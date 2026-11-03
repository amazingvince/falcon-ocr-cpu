//! The keys a prefill query tile sees, for every tile kernel: the parent's,
//! the FP32 static-shift tiles and the BF16 ones.
use super::Shape;

/// The span of the query tile starting at row `first_query`: its first
/// absolute position, the end of the keys its queries see (`[0,
/// visible_end)`) and whether every query is an image token (image rows see
/// every key below `image_end`: nothing to mask).
pub(super) fn tile_span(shape: &Shape, first_query: usize, queries: usize) -> (usize, usize, bool) {
    let first_absolute = shape.query_offset + first_query;
    let last_absolute = first_absolute + queries - 1;
    let intersects_image = first_absolute < shape.image_end && last_absolute >= shape.image_start;
    let visible_end = if intersects_image {
        (last_absolute + 1).max(shape.image_end)
    } else {
        last_absolute + 1
    };
    let all_image = first_absolute >= shape.image_start && last_absolute < shape.image_end;
    (first_absolute, visible_end, all_image)
}
