pub struct KeyPoint {
    pub x: usize,
    pub y: usize,
}

pub fn detect_features(image: &[u8], width: u32, height: u32, threshold: u8) -> Vec<KeyPoint> {
    let w = width as usize;
    let h = height as usize;
    let t = threshold as i16;
    let mut keypoints = Vec::new();

    // Offsets for the 16 pixels in the circle (radius 3)
    // 0 is top (0, -3), proceeding clockwise
    let offsets: [(i32, i32); 16] = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ];

    // Pre-calculate index offsets
    let pixel_offsets: Vec<isize> = offsets.iter()
        .map(|(dx, dy)| (dx + dy * (w as i32)) as isize)
        .collect();

    // Skip 3 pixels border to avoid boundary checks
    for y in 3..(h - 3) {
        for x in 3..(w - 3) {
            let p_idx = y * w + x;
            let p_val = image[p_idx] as i16;

            // Quick check: pixels 0, 4, 8, 12 (North, East, South, West)
            // Indices in offsets array: 0, 4, 8, 12
            // Corresponding pixel offsets: pixel_offsets[0], ...

            let v0 = image[(p_idx as isize + pixel_offsets[0]) as usize] as i16;
            let v4 = image[(p_idx as isize + pixel_offsets[4]) as usize] as i16;
            let v8 = image[(p_idx as isize + pixel_offsets[8]) as usize] as i16;
            let v12 = image[(p_idx as isize + pixel_offsets[12]) as usize] as i16;

            let mut darker_count = 0;
            let mut brighter_count = 0;

            if v0 < p_val - t { darker_count += 1; }
            if v0 > p_val + t { brighter_count += 1; }
            if v4 < p_val - t { darker_count += 1; }
            if v4 > p_val + t { brighter_count += 1; }
            if v8 < p_val - t { darker_count += 1; }
            if v8 > p_val + t { brighter_count += 1; }
            if v12 < p_val - t { darker_count += 1; }
            if v12 > p_val + t { brighter_count += 1; }

            // Optimization: FAST-9 requires at least 2 cardinals to be part of the segment
            if darker_count < 2 && brighter_count < 2 {
                continue;
            }

            // Full check
            // We need 9 contiguous pixels
            // This is a naive check for contiguous segment
            // We can optimize by checking the full ring

            // Collect ring values
            let ring: Vec<i16> = pixel_offsets.iter()
                .map(|&off| image[(p_idx as isize + off) as usize] as i16)
                .collect();

            if is_corner(&ring, p_val, t) {
                keypoints.push(KeyPoint { x, y });
            }
        }
    }

    keypoints
}

fn is_corner(ring: &[i16], p_val: i16, t: i16) -> bool {
    // Check for 9 contiguous pixels
    // Double the ring to handle wrap-around easily
    let extended_ring: Vec<i16> = ring.iter().chain(ring.iter()).cloned().collect();

    let lower = p_val - t;
    let upper = p_val + t;

    // Check brighter
    let mut contiguous = 0;
    for &val in &extended_ring {
        if val > upper {
            contiguous += 1;
        } else {
            contiguous = 0;
        }
        if contiguous >= 9 {
            return true;
        }
    }

    // Check darker
    contiguous = 0;
    for &val in &extended_ring {
        if val < lower {
            contiguous += 1;
        } else {
            contiguous = 0;
        }
        if contiguous >= 9 {
            return true;
        }
    }

    false
}
