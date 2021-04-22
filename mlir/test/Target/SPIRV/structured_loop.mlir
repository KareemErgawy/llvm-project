// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.GlobalVariable @GV1 bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  spv.GlobalVariable @GV2 bind(0, 1) : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  spv.func @loop_kernel() "None" {
    %0 = spv.mlir.addressof @GV1 : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
    %1 = spv.Constant 0 : i32
    %2 = spv.AccessChain %0[%1] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32
    %3 = spv.mlir.addressof @GV2 : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
    %5 = spv.AccessChain %3[%1] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, i32
    %6 = spv.Constant 4 : i32
    %7 = spv.Constant 42 : i32
    %8 = spv.Constant 2 : i32
    spv.Branch ^header(%6 : i32)

  ^header(%harg: i32):
    spv.mlir.structured_branch "Loop" ^entry(%harg : i32) [^merge, ^body]

  ^entry(%9: i32):
    %10 = spv.SLessThan %9, %7 : i32
    spv.BranchConditional %10, ^body(%9 : i32), ^merge

  ^body(%barg: i32):
    %11 = spv.AccessChain %2[%barg] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
    %12 = spv.Load "StorageBuffer" %11 : f32
    %13 = spv.AccessChain %5[%barg] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
    spv.Store "StorageBuffer" %13, %12 : f32
    %14 = spv.IAdd %barg, %8 : i32
    spv.Branch ^header(%14 : i32)

  ^merge:
    spv.Return
  }
  spv.EntryPoint "GLCompute" @loop_kernel
  spv.ExecutionMode @loop_kernel "LocalSize", 1, 1, 1
}

// CHECK:      spv.Branch ^bb1(%6 : i32)
// CHECK:    ^bb1(%7: i32):
// CHECK:      spv.mlir.loop {
// CHECK:        spv.Branch ^bb1(%7 : i32)
// CHECK:      ^bb1(%8: i32):
// CHECK:        spv.Branch ^bb2(%8 : i32)
// CHECK:      ^bb2(%9: i32):
// CHECK:        %10 = spv.Constant 42 : i32
// CHECK:        %11 = spv.SLessThan %9, %10 : i32
// CHECK:        spv.BranchConditional %11, ^bb3(%9 : i32), ^bb4
// CHECK:      ^bb3(%12: i32):
// CHECK:        %13 = spv.AccessChain %2[%12] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
// CHECK:        %14 = spv.Load "StorageBuffer" %13 : f32
// CHECK:        %15 = spv.AccessChain %5[%12] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
// CHECK:        spv.Store "StorageBuffer" %15, %14 : f32
// CHECK:        %16 = spv.Constant 2 : i32
// CHECK:        %17 = spv.IAdd %12, %16 : i32
// CHECK:        spv.Branch ^bb1(%17 : i32)
// CHECK:      ^bb4:
// CHECK:        spv.mlir.merge
// CHECK:      }
// CHECK:      spv.Return
