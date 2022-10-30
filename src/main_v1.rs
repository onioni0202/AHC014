// 戦略：ランダムに描ける長方形を選択して描けなくなるまで描く．それを制限時間までひたすら動かして最も良いスコアのものを最終解法とする

use proconio::input;
use rand::prelude::*;
use std::collections::{BTreeSet, HashSet};
use std::convert::TryFrom;

#[derive(Debug, Clone)]
pub struct LazyBIT<T, F> {
    size: usize,
    data: Vec<T>,
    lazy: Vec<T>,
    identity: T,
    operation: F,
}

impl<T: Copy + Clone + num::Integer + TryFrom<usize>, F: Fn(T, T) -> T> LazyBIT<T, F> {
    pub fn new(n: usize, id: T, op: F) -> LazyBIT<T, F> {
        let mut size = 1;
        while size < n {
            size <<= 1;
        }
        LazyBIT {
            size: size,
            data: vec![id; size * 2],
            lazy: vec![id; size * 2],
            identity: id,
            operation: op,
        }
    }

    //[a b)
    pub fn update(&mut self, a: usize, b: usize, x: T) {
        self.update_sub(0, 0, self.size, a, b, x);
    }

    //[a b)
    pub fn query(&mut self, a: usize, b: usize) -> T {
        self.query_sub(0, 0, self.size, a, b)
    }

    fn eval(&mut self, k: usize) {
        if self.lazy[k].eq(&self.identity) {
            return;
        }
        if k < self.size - 1 {
            self.lazy[k * 2 + 1] = self.lazy[k * 2 + 1] + self.lazy[k] / (T::one() + T::one());
            self.lazy[k * 2 + 2] = self.lazy[k * 2 + 2] + self.lazy[k] / (T::one() + T::one());
        }
        self.data[k] = (self.operation)(self.data[k], self.lazy[k]);
        self.lazy[k] = self.identity;
    }

    fn update_sub(&mut self, k: usize, l: usize, r: usize, a: usize, b: usize, x: T) {
        self.eval(k);
        if a <= l && r <= b {
            match T::try_from(r - l) {
                Ok(diff) => {
                    self.lazy[k] = (self.operation)(self.lazy[k], diff * x);
                }
                Err(_) => {
                    eprintln!("cannot convert from usize to T");
                }
            }
            self.eval(k);
        } else if a < r && l < b {
            self.update_sub(2 * k + 1, l, (l + r) / 2, a, b, x);
            self.update_sub(2 * k + 2, (l + r) / 2, r, a, b, x);
            self.data[k] = (self.operation)(self.data[2 * k + 1], self.data[2 * k + 2]);
        }
    }

    fn query_sub(&mut self, k: usize, l: usize, r: usize, a: usize, b: usize) -> T {
        self.eval(k);
        if r <= a || b <= l {
            return self.identity;
        } else if a <= l && r <= b {
            return self.data[k];
        } else {
            let vl = self.query_sub(2 * k + 1, l, (l + r) / 2, a, b);
            let vr = self.query_sub(2 * k + 2, (l + r) / 2, r, a, b);
            return (self.operation)(vl, vr);
        }
    }
}

#[derive(Debug, Clone)]
struct Grid {
    n: usize, // grid size
    coeff_score: f32,
    is_drawn_points: Vec<Vec<bool>>,
    is_drawn_0_deg_edges: Vec<LazyBIT<i32, fn(i32,i32)->i32>>,
    is_drawn_90_deg_edges: Vec<LazyBIT<i32, fn(i32,i32)->i32>>,
    is_drawn_45_deg_edges: Vec<LazyBIT<i32, fn(i32,i32)->i32>>,
    is_drawn_135_deg_edges: Vec<LazyBIT<i32, fn(i32,i32)->i32>>,
    line_0_deg: Vec<BTreeSet<usize>>,   // y = const
    line_90_deg: Vec<BTreeSet<usize>>,  // x = const
    line_45_deg: Vec<BTreeSet<usize>>,  // x+y = const
    line_135_deg: Vec<BTreeSet<usize>>, // x-y+n = const
}

impl Grid {
    fn draw_0_deg_rotated_rect(&mut self, rect_points: &Vec<(usize, usize)>) {
        let (x, y) = rect_points[0];
        assert!(self.is_drawn_points[x][y] == false);
        self.is_drawn_points[x][y] = true;
        self.line_0_deg[y].insert(x);
        self.line_90_deg[x].insert(y);
        self.line_45_deg[x - y + self.n].insert(x);
        self.line_135_deg[x + y].insert(x);
        for i in 0..4 {
            let (x1, y1) = rect_points[i];
            let (x2, y2) = rect_points[(i+1)%4];
            if x1 == x2 {
                self.is_drawn_90_deg_edges[x1].update(usize::min(y1, y2), usize::max(y1, y2), 1)
            } else {
                self.is_drawn_0_deg_edges[y1].update(usize::min(x1, x2), usize::max(x1, x2), 1)
            }
        }
    }

    fn draw_45_deg_rotated_rect(&mut self, rect_points: &Vec<(usize, usize)>) {
        let (x, y) = rect_points[0];
        assert!(self.is_drawn_points[x][y] == false);
        self.is_drawn_points[x][y] = true;
        self.line_0_deg[y].insert(x);
        self.line_90_deg[x].insert(y);
        self.line_45_deg[x - y + self.n].insert(x);
        self.line_135_deg[x + y].insert(x);
        for i in 0..4 {
            let (x1, y1) = rect_points[i];
            let (x2, y2) = rect_points[(i+1)%4];
            if x1-y1+self.n == x2-y2+self.n {
                self.is_drawn_45_deg_edges[x1-y1+self.n].update(usize::min(x1, x2), usize::max(x1, x2), 1)
            } else {
                self.is_drawn_135_deg_edges[x1+y1].update(usize::min(x1, x2), usize::max(x1, x2), 1)
            }
        }
    }

    fn can_draw_0_deg_rotated_rect(&mut self, rect_points: &Vec<(usize, usize)>) -> bool {
        // a first point should not be written
        if self.is_drawn_points[rect_points[0].0][rect_points[0].1] { 
            return false;
        }
        // any edges don't duplicate
        for i in 0..4 {
            let (x1, y1) = rect_points[i];
            let (x2, y2) = rect_points[(i+1)%4];
            if x1 == x2 && self.is_drawn_90_deg_edges[x1].query(usize::min(y1, y2), usize::max(y1, y2)) > 0 {
                return false;
            } 
            if y1 == y2 && self.is_drawn_0_deg_edges[y1].query(usize::min(x1, x2), usize::max(x1, x2)) > 0 {
                return false;
            }
        }
        // any other points don't exist on the edges
        for i in 0..4 {
            let (x, y) = rect_points[i];
            let mut corner_num = 0;
            if let Some(&upper) = self.line_90_deg[x].range((y+1)..).next() {
                if (x, upper) == rect_points[(i+1)%4] || (x, upper) == rect_points[(i+3)%4] {
                    corner_num += 1;
                }
            }
            if let Some(&lower) = self.line_90_deg[x].range(..y).next_back() {
                if (x, lower) == rect_points[(i+1)%4] || (x, lower) == rect_points[(i+3)%4] {
                    corner_num += 1;
                }
            }
            if let Some(&right) = self.line_0_deg[y].range((x+1)..).next() {
                if (right, y) == rect_points[(i+1)%4] || (right, y) == rect_points[(i+3)%4] {
                    corner_num += 1;
                }
            }
            if let Some(&left) = self.line_0_deg[y].range(..x).next_back() {
                if (left, y) == rect_points[(i+1)%4] || (left, y) == rect_points[(i+3)%4] {
                    corner_num += 1;
                }
            }
            if corner_num < 2 && (i % 2 == 0) {
                return false;
            }
            if corner_num < 1 && (i % 2 == 1) {
                return false;
            }
        }
        return true;
    }   
    
    fn can_draw_45_deg_rotated_rect(&mut self, rect_points: &Vec<(usize, usize)>) -> bool {
        // the first point should be in the grid
        if rect_points[0].0 >= self.n || rect_points[0].1 >= self.n {
            return false;
        }
        // the first point should be empty
        if self.is_drawn_points[rect_points[0].0][rect_points[0].1] { 
            return false;
        }
        // any edges don't duplicate
        for i in 0..4 {
            let (x1, y1) = rect_points[i];
            let (x2, y2) = rect_points[(i+1)%4];
            if x1-y1+self.n == x2-y2+self.n && 
            self.is_drawn_45_deg_edges[x1-y1+self.n].query(usize::min(x1, x2), usize::max(x1, x2)) > 0 {
                return false;
            } 
            if x1+y1 == x2+y2 && 
            self.is_drawn_135_deg_edges[x1+y1].query(usize::min(x1, x2), usize::max(x1, x2)) > 0 {
                return false;
            }
        }
        // any other points don't exist on the edges
        for i in 0..4 {
            let (x, y) = rect_points[i];
            let mut corner_num = 0;
            if let Some(&upper) = self.line_135_deg[x+y].range(..x).next_back() {
                if (upper, y-upper+x) == rect_points[(i+1)%4] || (upper, y-upper+x) == rect_points[(i+3)%4] {
                    corner_num += 1;
                }
            }
            if let Some(&lower) = self.line_135_deg[x+y].range((x+1)..).next() {
                if (lower, y-lower+x) == rect_points[(i+1)%4] || (lower, y-lower+x) == rect_points[(i+3)%4] {
                    corner_num += 1;
                }
            }
            if let Some(&right) = self.line_45_deg[x-y+self.n].range((x+1)..).next() {
                if (right, y+right-x) == rect_points[(i+1)%4] || (right, y+right-x) == rect_points[(i+3)%4] {
                    corner_num += 1;
                }
            }
            if let Some(&left) = self.line_45_deg[x-y+self.n].range(..x).next_back() {
                if (left, y+left-x) == rect_points[(i+1)%4] || (left, y+left-x) == rect_points[(i+3)%4] {
                    corner_num += 1;
                } 
            }
            if corner_num < 2 && (i % 2 == 0) {
                return false;
            }
            if corner_num < 1 && (i % 2 == 1) {
                return false;
            }
        }
        return true;
    }   

    fn calc_score(&self) -> f32 {
        let center = ((self.n-1)/2, (self.n-1)/2);
        let mut sum_weight = 0;
        for x in 0..self.n {
            for y in 0..self.n {
                if self.is_drawn_points[x][y] {
                    sum_weight += (x - center.0).pow(2) + (y - center.1).pow(2) + 1;
                }
            }
        }
        self.coeff_score * (sum_weight as f32)
    }
}


#[derive(Debug, Clone)]
struct DetectedRects {
    rects_vec: Vec<Vec<(usize, usize)>>,
    rects_set: HashSet<Vec<(usize, usize)>>,
    rng: rand_pcg::Pcg64Mcg
}

impl DetectedRects {
    fn new(seed: u128) -> DetectedRects {
        DetectedRects { 
            rects_vec: vec![], 
            rects_set: HashSet::new(), 
            rng: rand_pcg::Pcg64Mcg::new(seed)
        }
    }

    fn insert(&mut self, rect_points: Vec<(usize, usize)>) {
        if !self.rects_set.contains(&rect_points) {
            self.rects_vec.push(rect_points.clone());
            self.rects_set.insert(rect_points);
        }
    }

    fn change_seed(&mut self, new_seed: u128) {
        self.rng = rand_pcg::Pcg64Mcg::new(new_seed);
    }

    fn is_empty(&self) -> bool {
        self.rects_set.is_empty()
    }

    fn random_choice(&mut self) -> Option<Vec<(usize, usize)>> {
        if self.rects_vec.len() == 0 {
            return None;
        }
        let selected_index = self.rng.gen_range(0, self.rects_vec.len());
        let selected_rect = self.rects_vec.swap_remove(selected_index);
        self.rects_set.remove(&selected_rect);
        Some(selected_rect)
    }
}

fn input_grid_data() -> Grid {
    input! {
        n: usize,
        m: usize,
        xy: [(usize, usize); m]
    }
    let mut is_written_points = vec![vec![false; n]; n];
    let mut line_0 = vec![BTreeSet::new(); n];
    let mut line_90 = vec![BTreeSet::new(); n];
    let mut line_45 = vec![BTreeSet::new(); 2 * n];
    let mut line_135 = vec![BTreeSet::new(); 2 * n];
    let mut grid_weight = 0;
    for x in 0..n {
        for y in 0..n {
            grid_weight += (x - (n-1)/2).pow(2) + (y - (n-1)/2).pow(2) + 1;
        }
    }
    for &(x, y) in &xy {
        is_written_points[x][y] = true;
        line_0[y].insert(x);
        line_90[x].insert(y);
        line_45[x - y + n].insert(x);
        line_135[x + y].insert(x);
    }
    Grid {
        n,
        coeff_score: 1e6 * (n as f32) * (n as f32) / (m as f32) / (grid_weight as f32),
        is_drawn_points: is_written_points,
        is_drawn_0_deg_edges: vec![LazyBIT::new(n, 0, |x, y| x + y); n],
        is_drawn_90_deg_edges: vec![LazyBIT::new(n, 0, |x, y| x + y); n],
        is_drawn_45_deg_edges: vec![LazyBIT::new(n, 0, |x, y| x + y); 2 * n],
        is_drawn_135_deg_edges: vec![LazyBIT::new(n, 0, |x, y| x + y); 2 * n],
        line_0_deg: line_0,
        line_90_deg: line_90,
        line_45_deg: line_45,
        line_135_deg: line_135,
    }
}

fn main() {
    let mut grid = input_grid_data();
    let mut detected_0_rotated_rects = DetectedRects::new(0);
    let mut detected_45_rotated_rects = DetectedRects::new(0);
    for x in 0..grid.n {
        for y in 0..grid.n {
            if grid.is_drawn_points[x][y] {
                detect_0_deg_rotated_rect(&mut grid, &mut detected_0_rotated_rects, x, y);
                detect_45_deg_rotated_rect(&mut grid, &mut detected_45_rotated_rects, x, y);
            }
        }
    }
    //eprintln!("{:?}", detected_45_rotated_rects);
    let solution = greedy(grid, detected_0_rotated_rects, detected_45_rotated_rects, 4.9);
    print_solution(solution);
}

fn greedy(
    init_grid: Grid, 
    init_detected_0_rotated_rects: DetectedRects,
    init_detected_45_rotated_rects: DetectedRects,
    duration: f32
) -> Vec<Vec<(usize, usize)>> {
    let start_time = std::time::Instant::now();

    let mut best_score = 0.0;
    let mut best_solution = vec![];

    let mut iter_num:u128 = 0;
    
    while (std::time::Instant::now() - start_time).as_secs_f32() < duration {
        let mut grid = init_grid.clone();
        let mut detected_0_rotated_rects = init_detected_0_rotated_rects.clone();
        let mut detected_45_rotated_rects = init_detected_45_rotated_rects.clone();

        detected_0_rotated_rects.change_seed(iter_num);
        detected_45_rotated_rects.change_seed(iter_num);

        let mut solution = vec![];
        let mut rng = rand_pcg::Pcg64Mcg::new(iter_num);

        while !(detected_0_rotated_rects.is_empty() && detected_45_rotated_rects.is_empty()) {
            if rng.gen::<f32>() < 0.5 {
                if let Some(selected_rect) = detected_0_rotated_rects.random_choice() {
                    if grid.can_draw_0_deg_rotated_rect(&selected_rect) { 
                        grid.draw_0_deg_rotated_rect(&selected_rect);
                        detect_0_deg_rotated_rect(&mut grid, &mut detected_0_rotated_rects, selected_rect[0].0, selected_rect[0].1);
                        detect_45_deg_rotated_rect(&mut grid, &mut detected_45_rotated_rects, selected_rect[0].0, selected_rect[0].1);
                        solution.push(selected_rect);
                    }
                }
            } else {
                if let Some(selected_rect) = detected_45_rotated_rects.random_choice() { 
                    if grid.can_draw_45_deg_rotated_rect(&selected_rect) {
                        grid.draw_45_deg_rotated_rect(&selected_rect);
                        detect_0_deg_rotated_rect(&mut grid, &mut detected_0_rotated_rects, selected_rect[0].0, selected_rect[0].1);
                        detect_45_deg_rotated_rect(&mut grid, &mut detected_45_rotated_rects, selected_rect[0].0, selected_rect[0].1);
                        solution.push(selected_rect);
                    }
                }
            }
        }

        let now_score = grid.calc_score();
        if best_score < now_score {
            best_score = now_score;
            best_solution = solution.clone();
        }
        iter_num += 1;      
    }

    eprintln!("ITER: {}", iter_num);
    eprintln!("BEST_SCORE: {}", best_score);
    best_solution
}

fn print_solution(solution: Vec<Vec<(usize, usize)>>) {
    println!("{}", solution.len());
    for rect in solution {
        for i in 0..3{
            print!("{} {} ", rect[i].0, rect[i].1);
        }
        println!("{} {}", rect[3].0, rect[3].1);
    }
}


fn detect_0_deg_rotated_rect(
    grid: &mut Grid,
    detected_rects: &mut DetectedRects,
    x: usize,
    y: usize,
) {
    //left -> (upper or lower)
    if let Some(&left) = grid.line_0_deg[y].range(..x).next_back() {
        if let Some(&upper) = grid.line_90_deg[left].range((y + 1)..).next() {
            let rect_points = vec![(x, upper), (left, upper), (left, y), (x, y)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
        if let Some(&lower) = grid.line_90_deg[left].range(..y).next_back() {
            let rect_points = vec![(x, lower), (x, y), (left, y), (left, lower)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }

    //right -> (upper or lower)
    if let Some(&right) = grid.line_0_deg[y].range((x + 1)..).next() {
        if let Some(&upper) = grid.line_90_deg[right].range((y + 1)..).next() {
            let rect_points = vec![(x, upper), (x, y), (right, y), (right, upper)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
        if let Some(&lower) = grid.line_90_deg[right].range(..y).next_back() {
            let rect_points = vec![(x, lower), (right, lower), (right, y), (x, y)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }

    //lower -> (right or left)
    if let Some(&lower) = grid.line_90_deg[x].range(..y).next_back() {
        if let Some(&right) = grid.line_0_deg[lower].range((x + 1)..).next() {
            let rect_points = vec![(right, y), (x, y), (x, lower), (right, lower)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・ - ・
        //|
        //・
        if let Some(&right) = grid.line_0_deg[y].range((x + 1)..).next() {
            let rect_points = vec![(right, lower), (right, y), (x, y), (x, lower)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }


        if let Some(&left) = grid.line_0_deg[lower].range(..x).next_back() {
            let rect_points = vec![(left, y), (left, lower), (x, lower), (x, y)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・ - ・
        //     |
        //     ・
        if let Some(&left) = grid.line_0_deg[y].range(..x).next_back() {
            let rect_points = vec![(left, lower), (x, lower), (x, y), (left, y)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }
    

    //upper -> (right or left)
    if let Some(&upper) = grid.line_90_deg[x].range((y + 1)..).next() {
        if let Some(&right) = grid.line_0_deg[upper].range((x + 1)..).next() {
            let rect_points = vec![(right, y), (right, upper), (x, upper), (x, y)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・
        //|
        //・ - ・
        if let Some(&right) = grid.line_0_deg[y].range((x + 1)..).next() {
            let rect_points = vec![(right, upper), (x, upper), (x, y), (right, y)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        if let Some(&left) = grid.line_0_deg[upper].range(..x).next_back() {
            let rect_points = vec![(left, y), (x, y), (x, upper), (left, upper)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //     ・
        //     |
        //・ - ・
        if let Some(&left) = grid.line_0_deg[y].range(..x).next_back() {
            let rect_points = vec![(left, upper), (left, y), (x, y), (x, upper)];
            if grid.can_draw_0_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
        
    }
}

fn detect_45_deg_rotated_rect(
    grid: &mut Grid,
    detected_rects: &mut DetectedRects,
    x: usize,
    y: usize,
) {
    let n = grid.n;
    //left -> (upper or lower)
    if let Some(&left) = grid.line_45_deg[x - y + n].range(..x).next_back() {
        if let Some(&lower) = grid.line_135_deg[2*left+y-x].range((left + 1)..).next() {
            let rect_points = vec![
                (x+lower-left, y+left-lower), 
                (x, y), 
                (left, y+left-x), 
                (lower, y+2*left-lower-x)
            ];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
        if let Some(&upper) = grid.line_135_deg[2*left+y-x].range(..left).next_back() {
            let rect_points = vec![
                (x+upper-left, y-upper+left), 
                (upper, y+2*left-upper-x), 
                (left, y+left-x), 
                (x, y)
            ];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }

    //right -> (upper or lower)
    if let Some(&right) = grid.line_45_deg[x - y + n].range((x + 1)..).next() {
        if let Some(&lower) = grid.line_135_deg[2*right+y-x].range((right + 1)..).next() {
            let rect_points = vec![
                (x+lower-right, y-lower+right), 
                (lower, y+2*right-lower-x),
                (right, y+right-x),
                (x, y)
            ];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            } 
        }
        if let Some(&upper) = grid.line_135_deg[2*right+y-x].range(..right).next_back() {
            let rect_points = vec![
                (x+upper-right, y-upper+right), 
                (x, y),
                (right, y+right-x),
                (upper, y+2*right-upper-x)
            ];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }

    //lower -> (right or left)
    if let Some(&lower) = grid.line_135_deg[x+y].range((x+1)..).next() {
        if let Some(&left) = grid.line_45_deg[2*lower-y-x+n].range(..lower).next_back() {
            let rect_points = vec![
                (x-lower+left, y-lower+left), 
                (left, y+x+left-2*lower), 
                (lower, y+x-lower), 
                (x, y)];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・ - ・
        //     |
        //     ・
        if let Some(&left) = grid.line_45_deg[n+x-y].range(..x).next_back() {
            let rect_points = vec![
                (left+lower-x, left-lower+y), 
                (lower, y+x-lower), 
                (x, y), 
                (left, y-x+left)];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            } 
        }


        if let Some(&right) = grid.line_45_deg[2*lower-y-x+n].range((lower+1)..).next() {
            let rect_points = vec![
                (x+right-lower, y+right-lower), 
                (x, y), 
                (lower, y+x-lower), 
                (right, y+x+right-2*lower)];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・ - ・
        //|
        //・
        if let Some(&right) = grid.line_45_deg[n+x-y].range((x+1)..).next() {
            let rect_points = vec![
                (right+lower-x, right-lower+y), 
                (right, y+right-x), 
                (x, y), 
                (lower, y+x-lower)];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }
    

    //upper -> (right or left)
    if let Some(&upper) = grid.line_135_deg[x+y].range(..x).next_back() {
        if let Some(&left) = grid.line_45_deg[n+2*upper-y-x].range(..upper).next_back() {
            let rect_points = vec![
                (x-upper+left, y-upper+left), 
                (x, y), 
                (upper, x+y-upper), 
                (left, x+y-2*upper+left)];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //     ・
        //     |
        //・ - ・
        if let Some(&left) = grid.line_45_deg[x-y+n].range(..x).next_back() {
            let rect_points = vec![
                (upper+left-x, left-upper+y), 
                (left, y-x+left), 
                (x, y), 
                (upper, y+x-upper)];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        if let Some(&right) = grid.line_45_deg[n+2*upper-y-x].range((upper+1)..).next() {
            let rect_points = vec![
                (x+right-upper, y+right-upper), 
                (right, y+x-2*upper+right),
                (upper, y-upper+x), 
                (x, y)];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・
        //|
        //・ - ・
        if let Some(&right) = grid.line_45_deg[x-y+n].range((x+1)..).next() {
            let rect_points = vec![
                (upper+right-x, right-upper+y), 
                (upper, y+x-upper), 
                (x, y), 
                (right, y+upper-x)];
            if grid.can_draw_45_deg_rotated_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
        
    }
}

