//! openRTS's fork: opacity micromaps on Vulkan (VK_EXT_opacity_micromap;
//! crate::opacity_micromap).

use alloc::{sync::Arc, vec::Vec};
use core::cell::RefCell;
use core::sync::atomic::{AtomicBool, Ordering};

use ash::vk;

use crate::opacity_micromap::{bytes_per_micromap, OpacityMicromapDescriptor};

/// A built (or to be built, at its BLAS's first build) opacity micromap and
/// its geometry's indices.
pub struct OpacityMicromap {
    shared: Arc<super::DeviceShared>,
    pub(super) raw: vk::MicromapEXT,
    /// Its storage and the build's scratch (device memory).
    storage: vk::Buffer,
    scratch: vk::Buffer,
    device_memory: vk::DeviceMemory,
    /// The build's inputs and the BLAS's indices (host-visible memory).
    data: vk::Buffer,
    triangles: vk::Buffer,
    pub(super) indices: vk::Buffer,
    host_memory: vk::DeviceMemory,
    pub(super) usage: vk::MicromapUsageEXT,
    /// The geometry's triangles by state: (its micromaps' level, how many
    /// use a micromap), for the BLAS.
    pub(super) index_usage: vk::MicromapUsageEXT,
    micromap_count: u32,
    built: AtomicBool,
}

impl core::fmt::Debug for OpacityMicromap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OpacityMicromap").field("raw", &self.raw).finish()
    }
}

unsafe impl Send for OpacityMicromap {}
unsafe impl Sync for OpacityMicromap {}

impl Drop for OpacityMicromap {
    fn drop(&mut self) {
        let raw = &self.shared.raw;
        unsafe {
            if let Some(fns) = self.shared.extension_fns.opacity_micromap.as_ref() {
                (fns.fp().destroy_micromap_ext)(raw.handle(), self.raw, core::ptr::null());
            }
            for buffer in [self.storage, self.scratch, self.data, self.triangles, self.indices] {
                raw.destroy_buffer(buffer, None);
            }
            raw.free_memory(self.device_memory, None);
            raw.free_memory(self.host_memory, None);
        }
    }
}

impl OpacityMicromap {
    pub(super) fn address(&self, buffer: vk::Buffer) -> vk::DeviceAddress {
        unsafe {
            self.shared
                .raw
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        }
    }

    pub(super) fn build_info(&self) -> vk::MicromapBuildInfoEXT<'_> {
        vk::MicromapBuildInfoEXT::default()
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
            .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
            .mode(vk::BuildMicromapModeEXT::BUILD)
            .dst_micromap(self.raw)
            .usage_counts(core::slice::from_ref(&self.usage))
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: self.address(self.data),
            })
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.address(self.scratch),
            })
            .triangle_array(vk::DeviceOrHostAddressConstKHR {
                device_address: self.address(self.triangles),
            })
            .triangle_array_stride(size_of::<vk::MicromapTriangleEXT>() as u64)
    }

    /// Records its build once (the first BLAS build using it); true when it
    /// recorded one (a barrier is then due before the BLAS's build).
    pub(super) unsafe fn record_build(&self, command_buffer: vk::CommandBuffer) -> bool {
        if self.micromap_count == 0 || self.built.swap(true, Ordering::AcqRel) {
            return false;
        }
        let Some(fns) = self.shared.extension_fns.opacity_micromap.as_ref() else {
            return false;
        };
        let info = self.build_info();
        unsafe { (fns.fp().cmd_build_micromaps_ext)(command_buffer, 1, &info) };
        true
    }

    /// The BLAS geometry's link to it (pNext of its triangles).
    pub(super) fn triangles_link(&self) -> vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_> {
        vk::AccelerationStructureTrianglesOpacityMicromapEXT::default()
            .index_type(vk::IndexType::UINT32)
            .index_buffer(vk::DeviceOrHostAddressConstKHR {
                device_address: self.address(self.indices),
            })
            .index_stride(4)
            .base_triangle(0)
            .usage_counts(core::slice::from_ref(&self.index_usage))
            .micromap(if self.micromap_count == 0 {
                vk::MicromapEXT::null()
            } else {
                self.raw
            })
    }
}

std::thread_local! {
    /// The micromaps of the next BLAS's geometries this thread creates.
    static NEXT_BLAS: RefCell<Option<Vec<Option<Arc<OpacityMicromap>>>>> = const { RefCell::new(None) };
}

/// The next BLAS created on this thread takes these (one a geometry, none:
/// without); `None` clears it.
pub fn set_next_blas_opacity_micromaps(micromaps: Option<Vec<Option<Arc<OpacityMicromap>>>>) {
    NEXT_BLAS.with(|next| *next.borrow_mut() = micromaps);
}

pub(super) fn next_blas_peek() -> Option<Vec<Option<Arc<OpacityMicromap>>>> {
    NEXT_BLAS.with(|next| next.borrow().clone())
}

pub(super) fn next_blas_take() -> Option<Vec<Option<Arc<OpacityMicromap>>>> {
    NEXT_BLAS.with(|next| next.borrow_mut().take())
}

fn memory_type(
    shared: &super::DeviceShared,
    bits: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let properties = unsafe {
        shared
            .instance
            .raw
            .get_physical_device_memory_properties(shared.physical_device)
    };
    (0..properties.memory_type_count).find(|&i| {
        bits & (1 << i) != 0
            && properties.memory_types[i as usize]
                .property_flags
                .contains(wanted)
    })
}

impl super::Device {
    /// Whether the device makes opacity micromaps.
    pub fn supports_opacity_micromaps(&self) -> bool {
        self.shared.extension_fns.opacity_micromap.is_some()
    }

    /// An opacity micromap for one BLAS geometry (built at its first build).
    pub unsafe fn create_opacity_micromap(
        &self,
        desc: &OpacityMicromapDescriptor,
    ) -> Result<Arc<OpacityMicromap>, crate::DeviceError> {
        let shared = &self.shared;
        let fns = shared
            .extension_fns
            .opacity_micromap
            .as_ref()
            .ok_or(crate::DeviceError::Unexpected)?;
        let raw = &shared.raw;
        let per = bytes_per_micromap(desc.subdivision_level);
        let count = desc.micromap_count;
        if desc.data.len() < per * count as usize {
            return Err(crate::DeviceError::Unexpected);
        }
        let format = vk::OpacityMicromapFormatEXT::TYPE_4_STATE.as_raw() as u32;
        let usage = vk::MicromapUsageEXT {
            count,
            subdivision_level: desc.subdivision_level,
            format,
        };
        let used = desc.indices.iter().filter(|&&i| i >= 0).count() as u32;
        let index_usage = vk::MicromapUsageEXT {
            count: used,
            subdivision_level: desc.subdivision_level,
            format,
        };
        let triangles: Vec<vk::MicromapTriangleEXT> = (0..count)
            .map(|i| vk::MicromapTriangleEXT {
                data_offset: (i as usize * per) as u32,
                subdivision_level: desc.subdivision_level as u16,
                format: format as u16,
            })
            .collect();

        // The sizes of its storage and the build's scratch.
        let mut sizes = vk::MicromapBuildSizesInfoEXT::default();
        if count > 0 {
            let info = vk::MicromapBuildInfoEXT::default()
                .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
                .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
                .mode(vk::BuildMicromapModeEXT::BUILD)
                .usage_counts(core::slice::from_ref(&usage));
            unsafe {
                (fns.fp().get_micromap_build_sizes_ext)(
                    raw.handle(),
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &info,
                    &mut sizes,
                )
            };
        }

        let make_buffer = |size: u64, usage: vk::BufferUsageFlags| unsafe {
            raw.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size.max(16))
                    .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(super::map_host_device_oom_and_ioca_err)
        };
        let storage = make_buffer(sizes.micromap_size, vk::BufferUsageFlags::MICROMAP_STORAGE_EXT)?;
        let scratch = make_buffer(sizes.build_scratch_size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let data = make_buffer(
            desc.data.len() as u64,
            vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT,
        )?;
        let triangle_bytes = (triangles.len() * size_of::<vk::MicromapTriangleEXT>()) as u64;
        let triangle_buffer = make_buffer(triangle_bytes, vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT)?;
        let index_bytes = (desc.indices.len() * 4) as u64;
        let indices = make_buffer(
            index_bytes,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        )?;

        // One allocation for the device's two, one for the host's three.
        let place = |buffers: &[vk::Buffer], wanted: vk::MemoryPropertyFlags, fallback: vk::MemoryPropertyFlags| -> Result<(vk::DeviceMemory, Vec<u64>, u64), crate::DeviceError> {
            let mut offsets = Vec::new();
            let mut size = 0u64;
            let mut bits = !0u32;
            for &buffer in buffers {
                let req = unsafe { raw.get_buffer_memory_requirements(buffer) };
                size = size.next_multiple_of(req.alignment.max(256));
                offsets.push(size);
                size += req.size;
                bits &= req.memory_type_bits;
            }
            let index = memory_type(shared, bits, wanted)
                .or_else(|| memory_type(shared, bits, fallback))
                .ok_or(crate::DeviceError::OutOfMemory)?;
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = unsafe {
                raw.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(size.max(256))
                        .memory_type_index(index)
                        .push_next(&mut flags),
                    None,
                )
            }
            .map_err(super::map_host_device_oom_err)?;
            for (&buffer, &offset) in buffers.iter().zip(&offsets) {
                unsafe { raw.bind_buffer_memory(buffer, memory, offset) }
                    .map_err(super::map_host_device_oom_and_ioca_err)?;
            }
            Ok((memory, offsets, size))
        };
        let (device_memory, _, _) = place(
            &[storage, scratch],
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::empty(),
        )?;
        let (host_memory, offsets, host_size) = place(
            &[data, triangle_buffer, indices],
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let mapped = raw
                .map_memory(host_memory, 0, host_size.max(256), vk::MemoryMapFlags::empty())
                .map_err(super::map_host_device_oom_err)? as *mut u8;
            core::ptr::copy_nonoverlapping(desc.data.as_ptr(), mapped.add(offsets[0] as usize), desc.data.len());
            core::ptr::copy_nonoverlapping(
                triangles.as_ptr() as *const u8,
                mapped.add(offsets[1] as usize),
                triangle_bytes as usize,
            );
            core::ptr::copy_nonoverlapping(
                desc.indices.as_ptr() as *const u8,
                mapped.add(offsets[2] as usize),
                index_bytes as usize,
            );
            raw.unmap_memory(host_memory);
        }

        let mut micromap = vk::MicromapEXT::null();
        if count > 0 {
            let info = vk::MicromapCreateInfoEXT::default()
                .buffer(storage)
                .offset(0)
                .size(sizes.micromap_size)
                .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP);
            let result = unsafe { (fns.fp().create_micromap_ext)(raw.handle(), &info, core::ptr::null(), &mut micromap) };
            if result != vk::Result::SUCCESS {
                return Err(crate::DeviceError::OutOfMemory);
            }
        }
        Ok(Arc::new(OpacityMicromap {
            shared: Arc::clone(shared),
            raw: micromap,
            storage,
            scratch,
            device_memory,
            data,
            triangles: triangle_buffer,
            indices,
            host_memory,
            usage,
            index_usage,
            micromap_count: count,
            built: AtomicBool::new(count == 0),
        }))
    }
}
