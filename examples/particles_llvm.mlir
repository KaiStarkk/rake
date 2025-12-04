module {
  llvm.func @add(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.fadd %arg0, %arg1 : vector<8xf32>
    llvm.return %0 : vector<8xf32>
  }
  llvm.func @sub(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.fsub %arg0, %arg1 : vector<8xf32>
    llvm.return %0 : vector<8xf32>
  }
  llvm.func @mul(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.fmul %arg0, %arg1 : vector<8xf32>
    llvm.return %0 : vector<8xf32>
  }
  llvm.func @div(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.fdiv %arg0, %arg1 : vector<8xf32>
    llvm.return %0 : vector<8xf32>
  }
  llvm.func @safe_sqrt(%arg0: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.fcmp "oge" %arg0, %0 : vector<8xf32>
    %2 = llvm.intr.sqrt(%arg0) : (vector<8xf32>) -> vector<8xf32>
    %3 = llvm.select %1, %2, %0 : vector<8xi1>, vector<8xf32>
    llvm.return %3 : vector<8xf32>
  }
  llvm.func @magnitude(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.fmul %arg0, %arg0 : vector<8xf32>
    %1 = llvm.fmul %arg1, %arg1 : vector<8xf32>
    %2 = llvm.fmul %arg2, %arg2 : vector<8xf32>
    %3 = llvm.fadd %0, %1 : vector<8xf32>
    %4 = llvm.fadd %3, %2 : vector<8xf32>
    %5 = llvm.call @safe_sqrt(%4) : (vector<8xf32>) -> vector<8xf32>
    llvm.return %5 : vector<8xf32>
  }
  llvm.func @normalize_x(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.call @magnitude(%arg0, %arg1, %arg2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %1 = llvm.fdiv %arg0, %0 : vector<8xf32>
    llvm.return %1 : vector<8xf32>
  }
  llvm.func @normalize_y(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.call @magnitude(%arg0, %arg1, %arg2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %1 = llvm.fdiv %arg1, %0 : vector<8xf32>
    llvm.return %1 : vector<8xf32>
  }
  llvm.func @normalize_z(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.call @magnitude(%arg0, %arg1, %arg2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %1 = llvm.fdiv %arg2, %0 : vector<8xf32>
    llvm.return %1 : vector<8xf32>
  }
  llvm.func @clamp(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.fcmp "olt" %arg0, %arg1 : vector<8xf32>
    %1 = llvm.fcmp "ogt" %arg0, %arg2 : vector<8xf32>
    %2 = llvm.select %1, %arg2, %arg0 : vector<8xi1>, vector<8xf32>
    %3 = llvm.select %0, %arg1, %2 : vector<8xi1>, vector<8xf32>
    llvm.return %3 : vector<8xf32>
  }
  llvm.func @apply_gravity(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<1.600000e-02> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.mlir.constant(dense<9.810000e+00> : vector<8xf32>) : vector<8xf32>
    %2 = llvm.fmul %arg1, %1 : vector<8xf32>
    %3 = llvm.fmul %2, %0 : vector<8xf32>
    %4 = llvm.fsub %arg0, %3 : vector<8xf32>
    llvm.return %4 : vector<8xf32>
  }
  llvm.func @update_position(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<1.600000e-02> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.fmul %arg1, %0 : vector<8xf32>
    %2 = llvm.fadd %arg0, %1 : vector<8xf32>
    llvm.return %2 : vector<8xf32>
  }
  llvm.func @bounce_pos(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.fcmp "olt" %arg0, %0 : vector<8xf32>
    %2 = llvm.fsub %0, %arg0 : vector<8xf32>
    %3 = llvm.fcmp "ogt" %arg0, %arg2 : vector<8xf32>
    %4 = llvm.fsub %arg0, %arg2 : vector<8xf32>
    %5 = llvm.fsub %arg2, %4 : vector<8xf32>
    %6 = llvm.select %3, %5, %arg0 : vector<8xi1>, vector<8xf32>
    %7 = llvm.select %1, %2, %6 : vector<8xi1>, vector<8xf32>
    llvm.return %7 : vector<8xf32>
  }
  llvm.func @bounce_vel(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<8.000000e-01> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    %2 = llvm.fcmp "olt" %arg0, %1 : vector<8xf32>
    %3 = llvm.fmul %arg1, %0 : vector<8xf32>
    %4 = llvm.fsub %1, %3 : vector<8xf32>
    %5 = llvm.fcmp "ogt" %arg0, %arg2 : vector<8xf32>
    %6 = llvm.fmul %arg1, %0 : vector<8xf32>
    %7 = llvm.fsub %1, %6 : vector<8xf32>
    %8 = llvm.select %5, %7, %arg1 : vector<8xi1>, vector<8xf32>
    %9 = llvm.select %2, %4, %8 : vector<8xi1>, vector<8xf32>
    llvm.return %9 : vector<8xf32>
  }
  llvm.func @distance(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>, %arg3: vector<8xf32>, %arg4: vector<8xf32>, %arg5: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.fsub %arg3, %arg0 : vector<8xf32>
    %1 = llvm.fsub %arg4, %arg1 : vector<8xf32>
    %2 = llvm.fsub %arg5, %arg2 : vector<8xf32>
    %3 = llvm.call @magnitude(%0, %1, %2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    llvm.return %3 : vector<8xf32>
  }
  llvm.func @check_collision(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.mlir.constant(dense<1.000000e+00> : vector<8xf32>) : vector<8xf32>
    %2 = llvm.fadd %arg1, %arg2 : vector<8xf32>
    %3 = llvm.fcmp "olt" %arg0, %2 : vector<8xf32>
    %4 = llvm.select %3, %1, %0 : vector<8xi1>, vector<8xf32>
    llvm.return %4 : vector<8xf32>
  }
  llvm.func @fade(%arg0: vector<8xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.mlir.constant(dense<0.00999999977> : vector<8xf32>) : vector<8xf32>
    %2 = llvm.fcmp "ogt" %arg0, %1 : vector<8xf32>
    %3 = llvm.fmul %arg0, %arg1 : vector<8xf32>
    %4 = llvm.select %2, %3, %0 : vector<8xi1>, vector<8xf32>
    llvm.return %4 : vector<8xf32>
  }
  llvm.func @dot(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>, %arg3: vector<8xf32>, %arg4: vector<8xf32>, %arg5: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.fmul %arg0, %arg3 : vector<8xf32>
    %1 = llvm.fmul %arg1, %arg4 : vector<8xf32>
    %2 = llvm.fadd %0, %1 : vector<8xf32>
    %3 = llvm.fmul %arg2, %arg5 : vector<8xf32>
    %4 = llvm.fadd %2, %3 : vector<8xf32>
    llvm.return %4 : vector<8xf32>
  }
  llvm.func @reflect_x(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>, %arg3: vector<8xf32>, %arg4: vector<8xf32>, %arg5: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<2.000000e+00> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.call @dot(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %2 = llvm.fmul %1, %0 : vector<8xf32>
    %3 = llvm.fmul %2, %arg3 : vector<8xf32>
    %4 = llvm.fsub %arg0, %3 : vector<8xf32>
    llvm.return %4 : vector<8xf32>
  }
  llvm.func @reflect_y(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>, %arg3: vector<8xf32>, %arg4: vector<8xf32>, %arg5: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<2.000000e+00> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.call @dot(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %2 = llvm.fmul %1, %0 : vector<8xf32>
    %3 = llvm.fmul %2, %arg4 : vector<8xf32>
    %4 = llvm.fsub %arg1, %3 : vector<8xf32>
    llvm.return %4 : vector<8xf32>
  }
  llvm.func @reflect_z(%arg0: vector<8xf32>, %arg1: vector<8xf32>, %arg2: vector<8xf32>, %arg3: vector<8xf32>, %arg4: vector<8xf32>, %arg5: vector<8xf32>) -> vector<8xf32> {
    %0 = llvm.mlir.constant(dense<2.000000e+00> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.call @dot(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %2 = llvm.fmul %1, %0 : vector<8xf32>
    %3 = llvm.fmul %2, %arg5 : vector<8xf32>
    %4 = llvm.fsub %arg2, %3 : vector<8xf32>
    llvm.return %4 : vector<8xf32>
  }
}

