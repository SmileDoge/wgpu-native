//! openRTS's fork: opacity micromaps on D3D12 (DXR 1.2's opacity micromap
//! arrays, crate::opacity_micromap). `windows` 0.58 has none of DXR 1.2's
//! types: the few needed are laid out as d3d12.h (Agility SDK 1.618) has
//! them and put in the unions of the types it has.

use alloc::{sync::Arc, vec::Vec};
use core::cell::RefCell;
use core::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::{Direct3D12, Dxgi};

use crate::opacity_micromap::{bytes_per_micromap, OpacityMicromapDescriptor};

/// D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY.
const TYPE_OPACITY_MICROMAP_ARRAY: i32 = 2;
/// D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES.
const GEOMETRY_TYPE_OMM_TRIANGLES: i32 = 2;
/// D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT_OC1_4_STATE.
const FORMAT_4_STATE: u32 = 2;
/// D3D12_RAYTRACING_TIER_1_2.
const RAYTRACING_TIER_1_2: i32 = 12;

/// D3D12_RAYTRACING_OPACITY_MICROMAP_DESC.
#[repr(C)]
#[derive(Clone, Copy)]
struct OmmDesc {
    byte_offset: u32,
    level_and_format: u32, // SubdivisionLevel : 16, Format : 16
}

/// D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY.
#[repr(C)]
#[derive(Clone, Copy)]
struct HistogramEntry {
    count: u32,
    subdivision_level: u32,
    format: u32,
}

/// D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC.
#[repr(C)]
struct ArrayDesc {
    histogram_count: u32,
    histogram: *const HistogramEntry,
    input_buffer: u64,
    per_omm_descs: Direct3D12::D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE,
}

/// D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct OmmLinkage {
    index_buffer: Direct3D12::D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE,
    index_format: Dxgi::Common::DXGI_FORMAT,
    base_location: u32,
    array: u64,
}

/// D3D12_RAYTRACING_GEOMETRY_OMM_TRIANGLES_DESC.
#[repr(C)]
struct OmmTriangles {
    triangles: *const Direct3D12::D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC,
    linkage: *const OmmLinkage,
}

/// A built (or to be built, at its BLAS's first build) opacity micromap
/// array and its geometry's indices.
pub struct OpacityMicromap {
    array: Option<Direct3D12::ID3D12Resource>,
    scratch: Option<Direct3D12::ID3D12Resource>,
    /// The build's inputs (the micromaps, their descs) and the BLAS's indices.
    upload: Direct3D12::ID3D12Resource,
    data_offset: u64,
    descs_offset: u64,
    indices_offset: u64,
    histogram: HistogramEntry,
    built: AtomicBool,
}

unsafe impl Send for OpacityMicromap {}
unsafe impl Sync for OpacityMicromap {}

impl core::fmt::Debug for OpacityMicromap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OpacityMicromap").finish()
    }
}

impl OpacityMicromap {
    fn upload_address(&self) -> u64 {
        unsafe { self.upload.GetGPUVirtualAddress() }
    }

    fn array_desc(&self) -> ArrayDesc {
        ArrayDesc {
            histogram_count: 1,
            histogram: &self.histogram,
            input_buffer: self.upload_address() + self.data_offset,
            per_omm_descs: Direct3D12::D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                StartAddress: self.upload_address() + self.descs_offset,
                StrideInBytes: size_of::<OmmDesc>() as u64,
            },
        }
    }

    /// Records its build once (the first BLAS build using it); true when it
    /// recorded one (a UAV barrier is then due before the BLAS's build).
    pub(super) unsafe fn record_build(&self, list: &Direct3D12::ID3D12GraphicsCommandList4) -> bool {
        let (Some(array), Some(scratch)) = (&self.array, &self.scratch) else {
            return false;
        };
        if self.built.swap(true, Ordering::AcqRel) {
            return false;
        }
        let array_desc = self.array_desc();
        let desc = Direct3D12::D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: unsafe { array.GetGPUVirtualAddress() },
            Inputs: inputs(&array_desc),
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: unsafe { scratch.GetGPUVirtualAddress() },
        };
        unsafe { list.BuildRaytracingAccelerationStructure(&desc, None) };
        true
    }

    pub(super) fn linkage(&self) -> OmmLinkage {
        OmmLinkage {
            index_buffer: Direct3D12::D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                StartAddress: self.upload_address() + self.indices_offset,
                StrideInBytes: 4,
            },
            index_format: Dxgi::Common::DXGI_FORMAT_R32_UINT,
            base_location: 0,
            array: self
                .array
                .as_ref()
                .map_or(0, |a| unsafe { a.GetGPUVirtualAddress() }),
        }
    }
}

fn inputs(array_desc: &ArrayDesc) -> Direct3D12::D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    let mut anonymous = Direct3D12::D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { InstanceDescs: 0 };
    // pOpacityMicromapArrayDesc: a pointer in the union.
    unsafe {
        core::ptr::write(
            (&mut anonymous as *mut _) as *mut *const ArrayDesc,
            array_desc as *const ArrayDesc,
        )
    };
    Direct3D12::D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: Direct3D12::D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE(TYPE_OPACITY_MICROMAP_ARRAY),
        Flags: Direct3D12::D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: 1,
        DescsLayout: Direct3D12::D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: anonymous,
    }
}

/// The OMM geometries' parts, kept until the build or the sizes' query
/// (made with the geometries' count: never moved).
pub(super) struct Links {
    triangles: Vec<Direct3D12::D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC>,
    linkages: Vec<OmmLinkage>,
}

impl Links {
    pub(super) fn with_capacity(count: usize) -> Self {
        Self {
            triangles: Vec::with_capacity(count),
            linkages: Vec::with_capacity(count),
        }
    }

    /// Turns the triangle geometries with a micromap into OMM triangles.
    pub(super) fn attach(
        &mut self,
        geometries: &mut [Direct3D12::D3D12_RAYTRACING_GEOMETRY_DESC],
        micromaps: &[Option<Arc<OpacityMicromap>>],
    ) {
        for (geometry, micromap) in geometries.iter_mut().zip(micromaps) {
            let Some(micromap) = micromap else { continue };
            if geometry.Type != Direct3D12::D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES
                || self.triangles.len() == self.triangles.capacity()
            {
                continue;
            }
            self.triangles.push(unsafe { geometry.Anonymous.Triangles });
            self.linkages.push(micromap.linkage());
            let omm = OmmTriangles {
                triangles: self.triangles.last().unwrap(),
                linkage: self.linkages.last().unwrap(),
            };
            geometry.Type = Direct3D12::D3D12_RAYTRACING_GEOMETRY_TYPE(GEOMETRY_TYPE_OMM_TRIANGLES);
            unsafe { core::ptr::write((&mut geometry.Anonymous as *mut _) as *mut OmmTriangles, omm) };
        }
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

fn buffer(
    device: &Direct3D12::ID3D12Device,
    size: u64,
    heap: Direct3D12::D3D12_HEAP_TYPE,
    flags: Direct3D12::D3D12_RESOURCE_FLAGS,
    state: Direct3D12::D3D12_RESOURCE_STATES,
) -> Result<Direct3D12::ID3D12Resource, crate::DeviceError> {
    let properties = Direct3D12::D3D12_HEAP_PROPERTIES {
        Type: heap,
        ..Default::default()
    };
    let desc = Direct3D12::D3D12_RESOURCE_DESC {
        Dimension: Direct3D12::D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: Direct3D12::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    };
    let mut resource = None;
    unsafe {
        device.CreateCommittedResource(
            &properties,
            Direct3D12::D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut resource,
        )
    }
    .map_err(|_| crate::DeviceError::OutOfMemory)?;
    resource.ok_or(crate::DeviceError::Unexpected)
}

impl super::Device {
    /// Whether the device makes opacity micromaps (DXR 1.2: the D3D12
    /// runtime's and the driver's).
    pub fn supports_opacity_micromaps(&self) -> bool {
        let mut options = Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        let ok = unsafe {
            self.raw.CheckFeatureSupport(
                Direct3D12::D3D12_FEATURE_D3D12_OPTIONS5,
                (&mut options as *mut Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS5).cast(),
                size_of::<Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS5>() as u32,
            )
        }
        .is_ok();
        ok && options.RaytracingTier.0 >= RAYTRACING_TIER_1_2
    }

    /// An opacity micromap array for one BLAS geometry (built at its first build).
    pub unsafe fn create_opacity_micromap(
        &self,
        desc: &OpacityMicromapDescriptor,
    ) -> Result<Arc<OpacityMicromap>, crate::DeviceError> {
        let per = bytes_per_micromap(desc.subdivision_level);
        let count = desc.micromap_count;
        if desc.data.len() < per * count as usize {
            return Err(crate::DeviceError::Unexpected);
        }
        let histogram = HistogramEntry {
            count,
            subdivision_level: desc.subdivision_level,
            format: FORMAT_4_STATE,
        };
        let descs: Vec<OmmDesc> = (0..count)
            .map(|i| OmmDesc {
                byte_offset: (i as usize * per) as u32,
                level_and_format: desc.subdivision_level | (FORMAT_4_STATE << 16),
            })
            .collect();
        let align = |x: u64| x.next_multiple_of(256);
        let data_offset = 0u64;
        let descs_offset = align(desc.data.len() as u64);
        let indices_offset = align(descs_offset + (descs.len() * size_of::<OmmDesc>()) as u64);
        let upload_size = indices_offset + (desc.indices.len() * 4) as u64;
        let upload = buffer(
            &self.raw,
            upload_size,
            Direct3D12::D3D12_HEAP_TYPE_UPLOAD,
            Direct3D12::D3D12_RESOURCE_FLAG_NONE,
            Direct3D12::D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        unsafe {
            let mut mapped = core::ptr::null_mut();
            upload
                .Map(0, None, Some(&mut mapped))
                .map_err(|_| crate::DeviceError::Unexpected)?;
            let mapped = mapped as *mut u8;
            core::ptr::copy_nonoverlapping(desc.data.as_ptr(), mapped.add(data_offset as usize), desc.data.len());
            core::ptr::copy_nonoverlapping(
                descs.as_ptr() as *const u8,
                mapped.add(descs_offset as usize),
                descs.len() * size_of::<OmmDesc>(),
            );
            core::ptr::copy_nonoverlapping(
                desc.indices.as_ptr() as *const u8,
                mapped.add(indices_offset as usize),
                desc.indices.len() * 4,
            );
            upload.Unmap(0, None);
        }

        let mut micromap = OpacityMicromap {
            array: None,
            scratch: None,
            upload,
            data_offset,
            descs_offset,
            indices_offset,
            histogram,
            built: AtomicBool::new(count == 0),
        };
        if count > 0 {
            let device5: Direct3D12::ID3D12Device5 = windows::core::Interface::cast(&self.raw)
                .map_err(|_| crate::DeviceError::Unexpected)?;
            let array_desc = micromap.array_desc();
            let mut info = Direct3D12::D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
            unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs(&array_desc), &mut info) };
            if info.ResultDataMaxSizeInBytes == 0 {
                return Err(crate::DeviceError::Unexpected);
            }
            micromap.array = Some(buffer(
                &self.raw,
                info.ResultDataMaxSizeInBytes,
                Direct3D12::D3D12_HEAP_TYPE_DEFAULT,
                Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                Direct3D12::D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            )?);
            micromap.scratch = Some(buffer(
                &self.raw,
                info.ScratchDataSizeInBytes,
                Direct3D12::D3D12_HEAP_TYPE_DEFAULT,
                Direct3D12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                Direct3D12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )?);
        }
        Ok(Arc::new(micromap))
    }
}
