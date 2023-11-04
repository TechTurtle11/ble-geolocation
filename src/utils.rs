pub mod constants;

pub struct Point {
    pub x : i32,
    pub y: i32
}

impl Point {

    pub fn hash(self) -> f64 {
        ((self.x.pow(2) + self.y.pow(2)) as f64).sqrt()
    }
}