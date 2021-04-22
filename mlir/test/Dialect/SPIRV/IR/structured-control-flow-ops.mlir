// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

// Single loop

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // for (int i = 0; i < count; ++i) {}
  spv.func @loop(%count : i32) -> () "None" {
    %zero = spv.Constant 0: i32
    %one = spv.Constant 1: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

// CHECK:          spv.mlir.structured_branch "Loop" ^bb1 [^bb4, ^bb3]
    spv.mlir.structured_branch "Loop" ^entry [^merge, ^continue]

// CHECK-NEXT:   ^bb1:
  ^entry:
// CHECK-NEXT:     spv.Load
    %val0 = spv.Load "Function" %var : i32
// CHECK-NEXT:     spv.SLessThan
    %cmp = spv.SLessThan %val0, %count : i32
// CHECK-NEXT:     spv.BranchConditional %{{.*}} [1, 1], ^bb2, ^bb4
    spv.BranchConditional %cmp [1, 1], ^body, ^merge

// CHECK-NEXT:   ^bb2:
  ^body:
    // Do nothing
// CHECK-NEXT:     spv.Branch ^bb3
    spv.Branch ^continue

// CHECK-NEXT:   ^bb3:
  ^continue:
// CHECK-NEXT:     spv.Load
    %val1 = spv.Load "Function" %var : i32
// CHECK-NEXT:     spv.IAdd
    %add = spv.IAdd %val1, %one : i32
// CHECK-NEXT:     spv.Store
    spv.Store "Function" %var, %add : i32
// CHECK-NEXT:     spv.Branch ^bb1
    spv.Branch ^entry

// CHECK-NEXT:   ^bb4:
  ^merge:
    spv.Return
  }

  spv.func @main() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
}

