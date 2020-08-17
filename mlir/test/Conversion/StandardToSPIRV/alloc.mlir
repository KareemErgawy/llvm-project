// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -convert-std-to-spirv -canonicalize -verify-diagnostics %s -o - | FileCheck %s

//===----------------------------------------------------------------------===//
// std allocation/deallocation ops
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
  }
{
  func @alloc_dealloc_workgroup_mem(%arg0 : index, %arg1 : index) {
    %0 = alloc() : memref<4x5xf32, 3>
    %1 = load %0[%arg0, %arg1] : memref<4x5xf32, 3>
    store %1, %0[%arg0, %arg1] : memref<4x5xf32, 3>
    dealloc %0 : memref<4x5xf32, 3>
    return
  }
}
//     CHECK: spv.globalVariable @[[VAR:.+]] : !spv.ptr<!spv.struct<(!spv.array<20 x f32, stride=4>)>, Workgroup>
//     CHECK: func @alloc_dealloc_workgroup_mem
// CHECK-NOT:   alloc
//     CHECK:   %[[PTR:.+]] = spv._address_of @[[VAR]]
//     CHECK:   %[[LOADPTR:.+]] = spv.AccessChain %[[PTR]]
//     CHECK:   %[[VAL:.+]] = spv.Load "Workgroup" %[[LOADPTR]] : f32
//     CHECK:   %[[STOREPTR:.+]] = spv.AccessChain %[[PTR]]
//     CHECK:   spv.Store "Workgroup" %[[STOREPTR]], %[[VAL]] : f32
// CHECK-NOT:   dealloc
//     CHECK:   spv.Return

// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
  }
{
  func @alloc_dealloc_workgroup_mem(%arg0 : index, %arg1 : index) {
    %0 = alloc() : memref<4x5xi16, 3>
    %1 = load %0[%arg0, %arg1] : memref<4x5xi16, 3>
    store %1, %0[%arg0, %arg1] : memref<4x5xi16, 3>
    dealloc %0 : memref<4x5xi16, 3>
    return
  }
}

//       CHECK: spv.globalVariable @__workgroup_mem__{{[0-9]+}}
//  CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<20 x i32, stride=4>)>, Workgroup>
// CHECK_LABEL: spv.func @alloc_dealloc_workgroup_mem
//       CHECK:   %[[VAR:.+]] = spv._address_of @__workgroup_mem__0
//       CHECK:   %[[LOC:.+]] = spv.SDiv
//       CHECK:   %[[PTR:.+]] = spv.AccessChain %[[VAR]][%{{.+}}, %[[LOC]]]
//       CHECK:   %{{.+}} = spv.Load "Workgroup" %[[PTR]] : i32
//       CHECK:   %[[LOC:.+]] = spv.SDiv
//       CHECK:   %[[PTR:.+]] = spv.AccessChain %[[VAR]][%{{.+}}, %[[LOC]]]
//       CHECK:   %{{.+}} = spv.AtomicAnd "Workgroup" "AcquireRelease" %[[PTR]], %{{.+}} : !spv.ptr<i32, Workgroup>
//       CHECK:   %{{.+}} = spv.AtomicOr "Workgroup" "AcquireRelease" %[[PTR]], %{{.+}} : !spv.ptr<i32, Workgroup>


// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
  }
{
  func @two_allocs() {
    %0 = alloc() : memref<4x5xf32, 3>
    %1 = alloc() : memref<2x3xi32, 3>
    return
  }
}

//  CHECK-DAG: spv.globalVariable @__workgroup_mem__{{[0-9]+}}
// CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<6 x i32, stride=4>)>, Workgroup>
//  CHECK-DAG: spv.globalVariable @__workgroup_mem__{{[0-9]+}}
// CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<20 x f32, stride=4>)>, Workgroup>
//      CHECK: spv.func @two_allocs()
//      CHECK: spv.Return

// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
  }
{
  func @two_allocs_vector() {
    %0 = alloc() : memref<4xvector<4xf32>, 3>
    %1 = alloc() : memref<2xvector<2xi32>, 3>
    return
  }
}

//  CHECK-DAG: spv.globalVariable @__workgroup_mem__{{[0-9]+}}
// CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<2 x vector<2xi32>, stride=8>)>, Workgroup>
//  CHECK-DAG: spv.globalVariable @__workgroup_mem__{{[0-9]+}}
// CHECK-SAME:   !spv.ptr<!spv.struct<(!spv.array<4 x vector<4xf32>, stride=16>)>, Workgroup>
//      CHECK: spv.func @two_allocs_vector()
//      CHECK: spv.Return


// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
  }
{
  func @alloc_dealloc_dynamic_workgroup_mem(%arg0 : index) {
    // expected-error @+2 {{unhandled allocation type}}
    // expected-error @+1 {{'std.alloc' op operand #0 must be index}}
    %0 = alloc(%arg0) : memref<4x?xf32, 3>
    return
  }
}

// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
  }
{
  func @alloc_dealloc_mem() {
    // expected-error @+1 {{unhandled allocation type}}
    %0 = alloc() : memref<4x5xf32>
    return
  }
}


// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
  }
{
  func @alloc_dealloc_dynamic_workgroup_mem(%arg0 : memref<4x?xf32, 3>) {
    // expected-error @+2 {{unhandled deallocation type}}
    // expected-error @+1 {{'std.dealloc' op operand #0 must be memref of any type values}}
    dealloc %arg0 : memref<4x?xf32, 3>
    return
  }
}

// -----

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
  }
{
  func @alloc_dealloc_mem(%arg0 : memref<4x5xf32>) {
    // expected-error @+2 {{unhandled deallocation type}}
    // expected-error @+1 {{op operand #0 must be memref of any type values}}
    dealloc %arg0 : memref<4x5xf32>
    return
  }
}
