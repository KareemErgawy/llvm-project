// RUN: mlir-opt %s -split-input-file -pass-pipeline='func(canonicalize)' | FileCheck %s

spv.GlobalVariable @GV1 bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>

spv.GlobalVariable @GV2 bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>

func @loop_kernel() {
  %0 = spv.mlir.addressof @GV1 : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  %1 = spv.Constant 0 : i32
  %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32
  %3 = spv.mlir.addressof @GV2 : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  %5 = spv.AccessChain %3[%1] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32
  %6 = spv.Constant 4 : i32
  %7 = spv.Constant 42 : i32
  %8 = spv.Constant 2 : i32
  spv.mlir.loop {
    spv.Branch ^entry(%6 : i32)

  ^entry(%9: i32):
    %10 = spv.SLessThan %9, %7 : i32
    spv.BranchConditional %10, ^body, ^merge

  ^body:
    %11 = spv.AccessChain %2[%9] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
    %12 = spv.Load "StorageBuffer" %11 : f32
    %13 = spv.AccessChain %5[%9] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
    spv.Store "StorageBuffer" %13, %12 : f32
    %14 = spv.IAdd %9, %8 : i32
    spv.Branch ^entry(%14 : i32)

  ^merge:
    spv.mlir.merge
  }
  spv.Return
}

// CHECK:    spv.mlir.structured_branch "Loop" ^bb2(%1 : i32) [^bb5, ^bb3]
// CHECK:  ^bb2(%5: i32):
// CHECK:    %6 = spv.SLessThan %5, %2 : i32
// CHECK:    spv.BranchConditional %6, ^bb3(%5 : i32), ^bb5
// CHECK:  ^bb3(%7: i32):
// CHECK:    spv.Branch ^bb4
// CHECK:  ^bb4:
// CHECK:    %8 = spv.AccessChain %3[%0, %7] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
// CHECK:    %9 = spv.Load "StorageBuffer" %8 : f32
// CHECK:    %10 = spv.AccessChain %4[%0, %7] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
// CHECK:    spv.Store "StorageBuffer" %10, %9 : f32
// CHECK:    spv.Branch ^bb1
// CHECK:  ^bb5:
// CHECK:    spv.Branch ^bb6
