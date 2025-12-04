// Rake Language - Generated MLIR
// Target: 8-wide vectors (AVX2)
// Dialects: func, arith, vector, scf

module {
  
  // Module: Main
  
  // SoA type: Particle
  // Each field is a vector of 8 elements
  // !racks.soa_Particle = { x: f32, y: f32, z: f32, vx: f32, vy: f32, vz: f32, mass: f32 }
  
  // crunch function: add
  func.func @add(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
    %r1 = arith.addf %a, %b : vector<8xf32>
    func.return %r1 : vector<8xf32>
  }
  
  // crunch function: sub
  func.func @sub(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
    %r2 = arith.subf %a, %b : vector<8xf32>
    func.return %r2 : vector<8xf32>
  }
  
  // crunch function: mul
  func.func @mul(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
    %r3 = arith.mulf %a, %b : vector<8xf32>
    func.return %r3 : vector<8xf32>
  }
  
  // crunch function: div
  func.func @div(%a: vector<8xf32>, %b: vector<8xf32>) -> vector<8xf32> {
    %r4 = arith.divf %a, %b : vector<8xf32>
    func.return %r4 : vector<8xf32>
  }
  
  // rake function: safe_sqrt
  func.func @safe_sqrt(%x: vector<8xf32>) -> vector<8xf32> {
    %c5 = arith.constant 0.000000 : f32
    %splat6 = vector.splat %c5 : vector<8xf32>
    %cmp7 = arith.cmpf oge, %x, %splat6 : vector<8xf32>
    %call8 = math.sqrt %x : vector<8xf32>
    %true9 = arith.constant 1 : i1
    %splat10 = vector.splat %true9 : vector<8xi1>
    %c11 = arith.constant 0.000000 : f32
    %splat12 = vector.splat %c11 : vector<8xf32>
    %sel13 = arith.select %cmp7, %call8, %splat12 : vector<8xi1>, vector<8xf32>
    func.return %sel13 : vector<8xf32>
  }
  
  // crunch function: magnitude
  func.func @magnitude(%x: vector<8xf32>, %y: vector<8xf32>, %z: vector<8xf32>) -> vector<8xf32> {
    %r14 = arith.mulf %x, %x : vector<8xf32>
    // let x2 = %r14
    %r15 = arith.mulf %y, %y : vector<8xf32>
    // let y2 = %r15
    %r16 = arith.mulf %z, %z : vector<8xf32>
    // let z2 = %r16
    %r17 = arith.addf %r14, %r15 : vector<8xf32>
    %r18 = arith.addf %r17, %r16 : vector<8xf32>
    %call19 = func.call @safe_sqrt(%r18) : (vector<8xf32>) -> vector<8xf32>
    func.return %call19 : vector<8xf32>
  }
  
  // crunch function: normalize_x
  func.func @normalize_x(%x: vector<8xf32>, %y: vector<8xf32>, %z: vector<8xf32>) -> vector<8xf32> {
    %call20 = func.call @magnitude(%x, %y, %z) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    // let mag = %call20
    %r21 = arith.divf %x, %call20 : vector<8xf32>
    func.return %r21 : vector<8xf32>
  }
  
  // crunch function: normalize_y
  func.func @normalize_y(%x: vector<8xf32>, %y: vector<8xf32>, %z: vector<8xf32>) -> vector<8xf32> {
    %call22 = func.call @magnitude(%x, %y, %z) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    // let mag = %call22
    %r23 = arith.divf %y, %call22 : vector<8xf32>
    func.return %r23 : vector<8xf32>
  }
  
  // crunch function: normalize_z
  func.func @normalize_z(%x: vector<8xf32>, %y: vector<8xf32>, %z: vector<8xf32>) -> vector<8xf32> {
    %call24 = func.call @magnitude(%x, %y, %z) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    // let mag = %call24
    %r25 = arith.divf %z, %call24 : vector<8xf32>
    func.return %r25 : vector<8xf32>
  }
  
  // rake function: clamp
  func.func @clamp(%x: vector<8xf32>, %min: vector<8xf32>, %max: vector<8xf32>) -> vector<8xf32> {
    %cmp26 = arith.cmpf olt, %x, %min : vector<8xf32>
    %cmp27 = arith.cmpf ogt, %x, %max : vector<8xf32>
    %true28 = arith.constant 1 : i1
    %splat29 = vector.splat %true28 : vector<8xi1>
    %sel30 = arith.select %cmp27, %max, %x : vector<8xi1>, vector<8xf32>
    %sel31 = arith.select %cmp26, %min, %sel30 : vector<8xi1>, vector<8xf32>
    func.return %sel31 : vector<8xf32>
  }
  
  // crunch function: apply_gravity
  func.func @apply_gravity(%vy: vector<8xf32>, %mass: vector<8xf32>) -> vector<8xf32> {
    %c32 = arith.constant 9.810000 : f32
    %splat33 = vector.splat %c32 : vector<8xf32>
    // let gravity = %splat33
    %r34 = arith.mulf %splat33, %mass : vector<8xf32>
    %c35 = arith.constant 0.016000 : f32
    %splat36 = vector.splat %c35 : vector<8xf32>
    %r37 = arith.mulf %r34, %splat36 : vector<8xf32>
    %r38 = arith.subf %vy, %r37 : vector<8xf32>
    func.return %r38 : vector<8xf32>
  }
  
  // crunch function: update_position
  func.func @update_position(%pos: vector<8xf32>, %vel: vector<8xf32>) -> vector<8xf32> {
    %c39 = arith.constant 0.016000 : f32
    %splat40 = vector.splat %c39 : vector<8xf32>
    // let dt = %splat40
    %r41 = arith.mulf %vel, %splat40 : vector<8xf32>
    %r42 = arith.addf %pos, %r41 : vector<8xf32>
    func.return %r42 : vector<8xf32>
  }
  
  // rake function: bounce_pos
  func.func @bounce_pos(%pos: vector<8xf32>, %vel: vector<8xf32>, %limit: vector<8xf32>) -> vector<8xf32> {
    %c43 = arith.constant 0.000000 : f32
    %splat44 = vector.splat %c43 : vector<8xf32>
    %cmp45 = arith.cmpf olt, %pos, %splat44 : vector<8xf32>
    %c46 = arith.constant 0.000000 : f32
    %splat47 = vector.splat %c46 : vector<8xf32>
    %r48 = arith.subf %splat47, %pos : vector<8xf32>
    %cmp49 = arith.cmpf ogt, %pos, %limit : vector<8xf32>
    %r50 = arith.subf %pos, %limit : vector<8xf32>
    %r51 = arith.subf %limit, %r50 : vector<8xf32>
    %true52 = arith.constant 1 : i1
    %splat53 = vector.splat %true52 : vector<8xi1>
    %sel54 = arith.select %cmp49, %r51, %pos : vector<8xi1>, vector<8xf32>
    %sel55 = arith.select %cmp45, %r48, %sel54 : vector<8xi1>, vector<8xf32>
    func.return %sel55 : vector<8xf32>
  }
  
  // rake function: bounce_vel
  func.func @bounce_vel(%pos: vector<8xf32>, %vel: vector<8xf32>, %limit: vector<8xf32>) -> vector<8xf32> {
    %c56 = arith.constant 0.000000 : f32
    %splat57 = vector.splat %c56 : vector<8xf32>
    %cmp58 = arith.cmpf olt, %pos, %splat57 : vector<8xf32>
    %c59 = arith.constant 0.000000 : f32
    %splat60 = vector.splat %c59 : vector<8xf32>
    %c61 = arith.constant 0.800000 : f32
    %splat62 = vector.splat %c61 : vector<8xf32>
    %r63 = arith.mulf %vel, %splat62 : vector<8xf32>
    %r64 = arith.subf %splat60, %r63 : vector<8xf32>
    %cmp65 = arith.cmpf ogt, %pos, %limit : vector<8xf32>
    %c66 = arith.constant 0.000000 : f32
    %splat67 = vector.splat %c66 : vector<8xf32>
    %c68 = arith.constant 0.800000 : f32
    %splat69 = vector.splat %c68 : vector<8xf32>
    %r70 = arith.mulf %vel, %splat69 : vector<8xf32>
    %r71 = arith.subf %splat67, %r70 : vector<8xf32>
    %true72 = arith.constant 1 : i1
    %splat73 = vector.splat %true72 : vector<8xi1>
    %sel74 = arith.select %cmp65, %r71, %vel : vector<8xi1>, vector<8xf32>
    %sel75 = arith.select %cmp58, %r64, %sel74 : vector<8xi1>, vector<8xf32>
    func.return %sel75 : vector<8xf32>
  }
  
  // crunch function: distance
  func.func @distance(%x1: vector<8xf32>, %y1: vector<8xf32>, %z1: vector<8xf32>, %x2: vector<8xf32>, %y2: vector<8xf32>, %z2: vector<8xf32>) -> vector<8xf32> {
    %r76 = arith.subf %x2, %x1 : vector<8xf32>
    // let dx = %r76
    %r77 = arith.subf %y2, %y1 : vector<8xf32>
    // let dy = %r77
    %r78 = arith.subf %z2, %z1 : vector<8xf32>
    // let dz = %r78
    %call79 = func.call @magnitude(%r76, %r77, %r78) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    func.return %call79 : vector<8xf32>
  }
  
  // rake function: check_collision
  func.func @check_collision(%dist: vector<8xf32>, %radius: vector<8xf32>, %threshold: vector<8xf32>) -> vector<8xf32> {
    %r80 = arith.addf %radius, %threshold : vector<8xf32>
    %cmp81 = arith.cmpf olt, %dist, %r80 : vector<8xf32>
    %c82 = arith.constant 1.000000 : f32
    %splat83 = vector.splat %c82 : vector<8xf32>
    %true84 = arith.constant 1 : i1
    %splat85 = vector.splat %true84 : vector<8xi1>
    %c86 = arith.constant 0.000000 : f32
    %splat87 = vector.splat %c86 : vector<8xf32>
    %sel88 = arith.select %cmp81, %splat83, %splat87 : vector<8xi1>, vector<8xf32>
    func.return %sel88 : vector<8xf32>
  }
  
  // rake function: fade
  func.func @fade(%alpha: vector<8xf32>, %decay: vector<8xf32>) -> vector<8xf32> {
    %c89 = arith.constant 0.010000 : f32
    %splat90 = vector.splat %c89 : vector<8xf32>
    %cmp91 = arith.cmpf ogt, %alpha, %splat90 : vector<8xf32>
    %r92 = arith.mulf %alpha, %decay : vector<8xf32>
    %true93 = arith.constant 1 : i1
    %splat94 = vector.splat %true93 : vector<8xi1>
    %c95 = arith.constant 0.000000 : f32
    %splat96 = vector.splat %c95 : vector<8xf32>
    %sel97 = arith.select %cmp91, %r92, %splat96 : vector<8xi1>, vector<8xf32>
    func.return %sel97 : vector<8xf32>
  }
  
  // crunch function: dot
  func.func @dot(%x1: vector<8xf32>, %y1: vector<8xf32>, %z1: vector<8xf32>, %x2: vector<8xf32>, %y2: vector<8xf32>, %z2: vector<8xf32>) -> vector<8xf32> {
    %r98 = arith.mulf %x1, %x2 : vector<8xf32>
    %r99 = arith.mulf %y1, %y2 : vector<8xf32>
    %r100 = arith.addf %r98, %r99 : vector<8xf32>
    %r101 = arith.mulf %z1, %z2 : vector<8xf32>
    %r102 = arith.addf %r100, %r101 : vector<8xf32>
    func.return %r102 : vector<8xf32>
  }
  
  // crunch function: reflect_x
  func.func @reflect_x(%vx: vector<8xf32>, %vy: vector<8xf32>, %vz: vector<8xf32>, %nx: vector<8xf32>, %ny: vector<8xf32>, %nz: vector<8xf32>) -> vector<8xf32> {
    %call103 = func.call @dot(%vx, %vy, %vz, %nx, %ny, %nz) : (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    // let d = %call103
    %c104 = arith.constant 2.000000 : f32
    %splat105 = vector.splat %c104 : vector<8xf32>
    %r106 = arith.mulf %splat105, %call103 : vector<8xf32>
    %r107 = arith.mulf %r106, %nx : vector<8xf32>
    %r108 = arith.subf %vx, %r107 : vector<8xf32>
    func.return %r108 : vector<8xf32>
  }
  
  // crunch function: reflect_y
  func.func @reflect_y(%vx: vector<8xf32>, %vy: vector<8xf32>, %vz: vector<8xf32>, %nx: vector<8xf32>, %ny: vector<8xf32>, %nz: vector<8xf32>) -> vector<8xf32> {
    %call109 = func.call @dot(%vx, %vy, %vz, %nx, %ny, %nz) : (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    // let d = %call109
    %c110 = arith.constant 2.000000 : f32
    %splat111 = vector.splat %c110 : vector<8xf32>
    %r112 = arith.mulf %splat111, %call109 : vector<8xf32>
    %r113 = arith.mulf %r112, %ny : vector<8xf32>
    %r114 = arith.subf %vy, %r113 : vector<8xf32>
    func.return %r114 : vector<8xf32>
  }
  
  // crunch function: reflect_z
  func.func @reflect_z(%vx: vector<8xf32>, %vy: vector<8xf32>, %vz: vector<8xf32>, %nx: vector<8xf32>, %ny: vector<8xf32>, %nz: vector<8xf32>) -> vector<8xf32> {
    %call115 = func.call @dot(%vx, %vy, %vz, %nx, %ny, %nz) : (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    // let d = %call115
    %c116 = arith.constant 2.000000 : f32
    %splat117 = vector.splat %c116 : vector<8xf32>
    %r118 = arith.mulf %splat117, %call115 : vector<8xf32>
    %r119 = arith.mulf %r118, %nz : vector<8xf32>
    %r120 = arith.subf %vz, %r119 : vector<8xf32>
    func.return %r120 : vector<8xf32>
  }
}
