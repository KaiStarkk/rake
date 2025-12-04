// STUB: Particle simulation in Odin
// Odin has excellent SIMD support via simd package
//
// Key features for comparison:
// - Explicit SIMD types (#simd[8]f32)
// - SOA/AOS layout control
// - C-like performance with better ergonomics
//
// TODO:
// - Use #simd[8]f32 for particle positions/velocities
// - Implement SoA layout with Odin's SOA helper
// - Compare explicit SIMD vs Rake's implicit vectors

package main

import "core:fmt"

main :: proc() {
    fmt.println("Odin particles benchmark - not yet implemented")
}
