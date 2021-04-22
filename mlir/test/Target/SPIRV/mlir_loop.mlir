// TODO This is just to aid in development, will be properly updated as a test later.

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
// CHECK:        spv.Branch ^bb1(%{{.*}} : i32)
// CHECK-NEXT: ^bb1(%[[OUTARG:.*]]: i32):
// CHECK-NEXT:   spv.mlir.loop {
    spv.mlir.loop {
// CHECK-NEXT:     spv.Branch ^bb1(%[[OUTARG]] : i32)
      spv.Branch ^header(%6 : i32)
// CHECK-NEXT:   ^bb1(%[[HEADARG:.*]]: i32):
    ^header(%9: i32):
      %10 = spv.SLessThan %9, %7 : i32
// CHECK:          spv.BranchConditional %{{.*}}, ^bb2, ^bb3
      spv.BranchConditional %10, ^body, ^merge
// CHECK-NEXT:   ^bb2:     // pred: ^bb1
    ^body:
      %11 = spv.AccessChain %2[%9] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
      %12 = spv.Load "StorageBuffer" %11 : f32
      %13 = spv.AccessChain %5[%9] : !spv.ptr<!spv.array<10 x f32, stride=4>, StorageBuffer>, i32
      spv.Store "StorageBuffer" %13, %12 : f32
// CHECK:          %[[ADD:.*]] = spv.IAdd
      %14 = spv.IAdd %9, %8 : i32
// CHECK-NEXT:     spv.Branch ^bb1(%[[ADD]] : i32)
      spv.Branch ^header(%14 : i32)
// CHECK-NEXT:   ^bb3:
    ^merge:
// CHECK-NEXT:     spv.mlir.merge
      spv.mlir.merge
    }
    spv.Return
  }
  spv.EntryPoint "GLCompute" @loop_kernel
  spv.ExecutionMode @loop_kernel "LocalSize", 1, 1, 1
}
