// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s
// RUN: mlir-translate -test-spirv-roundtrip-deserialize-to-scf %s | FileCheck -check-prefix=CHECK-SCF %s
// RUN: mlir-translate -test-spirv-roundtrip-deserialize-to-scf-to-so %s | FileCheck -check-prefix=CHECK-SCF-SO %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  spv.func @loop(%count : i32) -> () "None" {
    %zero = spv.Constant 0: i32
    %one = spv.Constant 1: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>
    spv.Branch ^header

  ^header:
    spv.mlir.structured_branch "Loop" ^entry [^merge, ^continue]

  ^entry:
    %val0 = spv.Load "Function" %var : i32
    %cmp = spv.SLessThan %val0, %count : i32
    spv.BranchConditional %cmp [1, 1], ^body, ^merge

  ^body:
    // Do nothing
    spv.Branch ^continue

  ^continue:
    %val1 = spv.Load "Function" %var : i32
    %add = spv.IAdd %val1, %one : i32
    spv.Store "Function" %var, %add : i32
    spv.Branch ^header

  ^merge:
    spv.Return
  }

  spv.func @main() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
}

// Tests the path:
//   spv.mlir.structured_branch "Loop"
//             |
//             | Serialize
//             V
//   SPIR-V structured loop
//             |
//             | Deserialize
//             V
//   spv.mlir.loop
//
// CHECK:   spv.Constant 0 : i32
// CHECK:   spv.Variable init
// CHECK:   spv.Branch ^[[loop:.*]]
// CHECK: ^[[loop]]:
// CHECK:   spv.mlir.loop
// CHECK:     spv.Branch ^[[bb1:.*]]
// CHECK:   ^[[bb1]]:
// CHECK:     spv.Branch ^[[header:.*]]
// CHECK:   ^[[header]]:
// CHECK:     spv.Load
// CHECK:     spv.SLessThan
// CHECK:     spv.BranchConditional %{{.*}} [1, 1], ^[[bb2:.*]], ^[[merge:.*]]
// CHECK:   ^[[bb2]]:
// CHECK:     spv.Branch ^[[continue:.*]]
// CHECK:   ^[[continue]]:
// CHECK:     spv.Load
// CHECK:     spv.Constant 1
// CHECK:     spv.IAdd
// CHECK:     spv.Store
// CHECK:     spv.Branch ^[[bb1]]
// CHECK:   ^[[merge]]:
// CHECK:     spv.mlir.merge

// Tests the path:
//   spv.mlir.structured_branch "Loop"
//             |
//             | Serialize
//             V
//   SPIR-V structured loop
//             |
//             | Deserialize
//             V
//   spv.mlir.structured_branch "Loop"
//
// CHECK-SCF:      %0 = spv.Constant 0 : i32
// CHECK-SCF:      %1 = spv.Variable init(%0) : !spv.ptr<i32, Function>
// CHECK-SCF:      spv.Branch ^bb1
// CHECK-SCF:    ^bb1:
// CHECK-SCF:      spv.mlir.structured_branch "Loop" ^bb2 [^bb4, ^bb3]
// CHECK-SCF:    ^bb2:
// CHECK-SCF:      %2 = spv.Load "Function" %1 : i32
// CHECK-SCF:      %3 = spv.SLessThan %2, %arg0 : i32
// CHECK-SCF:      spv.BranchConditional %3 [1, 1], ^bb5, ^bb4
// CHECK-SCF:    ^bb3:
// CHECK-SCF:      %4 = spv.Load "Function" %1 : i32
// CHECK-SCF:      %5 = spv.Constant 1 : i32
// CHECK-SCF:      %6 = spv.IAdd %4, %5 : i32
// CHECK-SCF:      spv.Store "Function" %1, %6 : i32
// CHECK-SCF:      spv.Branch ^bb1
// CHECK-SCF:    ^bb4:
// CHECK-SCF:      spv.Return
// CHECK-SCF:    ^bb5:
// CHECK-SCF:      spv.Branch ^bb3


// Tests the path:
//   spv.mlir.structured_branch "Loop"
//             |
//             | Serialize
//             V
//   SPIR-V structured loop
//             |
//             | Deserialize
//             V
//   spv.mlir.structured_branch "Loop"
//             |
//             | Convert
//             V
//   spv.mlir.loop
//
// CHECK-SCF-SO:      spv.Branch ^bb1
// CHECK-SCF-SO:    ^bb1:  // pred: ^bb0
// CHECK-SCF-SO:      spv.mlir.loop {
// CHECK-SCF-SO:        spv.Branch ^bb1
// CHECK-SCF-SO:      ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK-SCF-SO:        spv.mlir.structured_branch "Loop" ^bb2 [^bb5, ^bb3]
// CHECK-SCF-SO:      ^bb2:  // pred: ^bb1
// CHECK-SCF-SO:        %2 = spv.Load "Function" %1 : i32
// CHECK-SCF-SO:        %3 = spv.SLessThan %2, %arg0 : i32
// CHECK-SCF-SO:        spv.BranchConditional %3 [1, 1], ^bb4, ^bb5
// CHECK-SCF-SO:      ^bb3:  // 2 preds: ^bb1, ^bb4
// CHECK-SCF-SO:        %4 = spv.Load "Function" %1 : i32
// CHECK-SCF-SO:        %5 = spv.Constant 1 : i32
// CHECK-SCF-SO:        %6 = spv.IAdd %4, %5 : i32
// CHECK-SCF-SO:        spv.Store "Function" %1, %6 : i32
// CHECK-SCF-SO:        spv.Branch ^bb1
// CHECK-SCF-SO:      ^bb4:  // pred: ^bb2
// CHECK-SCF-SO:        spv.Branch ^bb3
// CHECK-SCF-SO:      ^bb5:  // 2 preds: ^bb1, ^bb2
// CHECK-SCF-SO:        spv.mlir.merge
// CHECK-SCF-SO:      }
// CHECK-SCF-SO:      spv.Return
