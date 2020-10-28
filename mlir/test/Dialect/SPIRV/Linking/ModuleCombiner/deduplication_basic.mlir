// RUN: mlir-opt -test-spirv-module-combiner -split-input-file -verify-diagnostics %s | FileCheck %s

// Deduplicate 2 global variables with the same descriptor set and binding.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.globalVariable @foo
// CHECK-NEXT:     spv.func @use_foo
// CHECK-NEXT:       spv._address_of @foo
// CHECK-NEXT:       spv.Load
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:     spv.func @use_bar
// CHECK-NEXT:       spv._address_of @foo
// CHECK-NEXT:       spv.Load
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.globalVariable @foo bind(1, 0) : !spv.ptr<f32, Input>

  spv.func @use_foo() -> f32 "None" {
    %0 = spv._address_of @foo : !spv.ptr<f32, Input>
    %1 = spv.Load "Input" %0 : f32
    spv.ReturnValue %1 : f32
  }
}

spv.module Logical GLSL450 {
  spv.globalVariable @bar bind(1, 0) : !spv.ptr<f32, Input>

  spv.func @use_bar() -> f32 "None" {
    %0 = spv._address_of @bar : !spv.ptr<f32, Input>
    %1 = spv.Load "Input" %0 : f32
    spv.ReturnValue %1 : f32
  }
}
}

// -----

// Deduplicate 2 global variables with the same built-in attribute.

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.globalVariable @foo built_in("GlobalInvocationId")
// CHECK-NEXT:     spv.func @use_bar
// CHECK-NEXT:       spv._address_of @foo
// CHECK-NEXT:       spv.Load
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.globalVariable @foo built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
}

spv.module Logical GLSL450 {
  spv.globalVariable @bar built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>

  spv.func @use_bar() -> vector<3xi32> "None" {
    %0 = spv._address_of @bar : !spv.ptr<vector<3xi32>, Input>
    %1 = spv.Load "Input" %0 : vector<3xi32>
    spv.ReturnValue %1 : vector<3xi32>
  }
}
}

// -----

// CHECK:      module {
// CHECK-NEXT:   spv.module Logical GLSL450 {
// CHECK-NEXT:     spv.specConstant @foo spec_id(5)

// CHECK-NEXT:     spv.func @use_foo()
// CHECK-NEXT:       %0 = spv._reference_of @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }

// CHECK-NEXT:     spv.func @use_bar()
// CHECK-NEXT:       %0 = spv._reference_of @foo
// CHECK-NEXT:       spv.ReturnValue
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
spv.module Logical GLSL450 {
  spv.specConstant @foo spec_id(5) = 1. : f32

  spv.func @use_foo() -> (f32) "None" {
    %0 = spv._reference_of @foo : f32
    spv.ReturnValue %0 : f32
  }
}

spv.module Logical GLSL450 {
  spv.specConstant @bar spec_id(5) = 1. : f32

  spv.func @use_bar() -> (f32) "None" {
    %0 = spv._reference_of @bar : f32
    spv.ReturnValue %0 : f32
  }
}
}
