// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // for (int i = 0; i < count; ++i) {}
  spv.func @loop(%count : i32) -> () "None" {
    %zero = spv.Constant 0: i32
    %one = spv.Constant 1: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>
    spv.Branch ^header

  ^header:
// CHECK:          spv.mlir.structured_branch "Loop" ^bb2 [^bb5, ^bb4]
    spv.mlir.structured_branch "Loop" ^entry [^merge, ^continue]

// CHECK-NEXT:   ^bb2:
  ^entry:
// CHECK-NEXT:     spv.Load
    %val0 = spv.Load "Function" %var : i32
// CHECK-NEXT:     spv.SLessThan
    %cmp = spv.SLessThan %val0, %count : i32
// CHECK-NEXT:     spv.BranchConditional %{{.*}} [1, 1], ^bb3, ^bb5
    spv.BranchConditional %cmp [1, 1], ^body, ^merge

// CHECK-NEXT:   ^bb3:
  ^body:
// CHECK-NEXT:     spv.Branch ^bb4
    spv.Branch ^continue

// CHECK-NEXT:   ^bb4:
  ^continue:
// CHECK-NEXT:     spv.Load
    %val1 = spv.Load "Function" %var : i32
// CHECK-NEXT:     spv.IAdd
    %add = spv.IAdd %val1, %one : i32
// CHECK-NEXT:     spv.Store
    spv.Store "Function" %var, %add : i32
// CHECK-NEXT:     spv.Branch ^bb1
    spv.Branch ^header

// CHECK-NEXT:   ^bb5:
  ^merge:
    spv.Return
  }

  spv.func @main() -> () "None" {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
}

