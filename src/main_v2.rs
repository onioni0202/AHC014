// 戦略：最もスコアが高くなる点をBTreeSetにより取ってくる．

use proconio::input;
use std::collections::{BTreeSet, HashSet, BinaryHeap};
use std::convert::TryFrom;

//const INF: i32 = 10_000_000;

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
    m: usize,  // the number of initial points
    ratio: f32,
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
    fn draw_rect(&mut self, rect_points: &[(usize, usize); 4]) {
        if rect_points[0].0 == rect_points[1].0 || rect_points[0].1 == rect_points[1].1 {
            self.draw_0_deg_rotated_rect(rect_points);
        } else {
            self.draw_45_deg_rotated_rect(rect_points);
        }
    }

    fn delete_rect(&mut self, rect_points: &[(usize, usize); 4]) {
        if rect_points[0].0 == rect_points[1].0 || rect_points[0].1 == rect_points[1].1 {
            self.delete_0_deg_rotated_rect(rect_points);
        } else {
            self.delete_45_deg_rotated_rect(rect_points);
        }
    }

    fn can_draw_rect(&mut self, rect_points: &[(usize, usize); 4]) -> bool {
        if rect_points[0].0 == rect_points[1].0 || rect_points[0].1 == rect_points[1].1 {
            return self.can_draw_0_deg_rotated_rect(rect_points);
        } else {
            return self.can_draw_45_deg_rotated_rect(rect_points);
        }
    }

    fn draw_0_deg_rotated_rect(&mut self, rect_points: &[(usize, usize); 4]) {
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

    fn draw_45_deg_rotated_rect(&mut self, rect_points: &[(usize, usize); 4]) {
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

    fn delete_0_deg_rotated_rect(&mut self, rect_points: &[(usize, usize); 4]) {
        let (x, y) = rect_points[0];
        assert!(self.is_drawn_points[x][y] == true);
        self.is_drawn_points[x][y] = false;
        self.line_0_deg[y].remove(&x);
        self.line_90_deg[x].remove(&y);
        self.line_45_deg[x - y + self.n].remove(&x);
        self.line_135_deg[x + y].remove(&x);
        for i in 0..4 {
            let (x1, y1) = rect_points[i];
            let (x2, y2) = rect_points[(i+1)%4];
            if x1 == x2 {
                self.is_drawn_90_deg_edges[x1].update(usize::min(y1, y2), usize::max(y1, y2), -1)
            } else {
                self.is_drawn_0_deg_edges[y1].update(usize::min(x1, x2), usize::max(x1, x2), -1)
            }
        }
    }

    fn delete_45_deg_rotated_rect(&mut self, rect_points: &[(usize, usize); 4]) {
        let (x, y) = rect_points[0];
        assert!(self.is_drawn_points[x][y] == true);
        self.is_drawn_points[x][y] = false;
        self.line_0_deg[y].remove(&x);
        self.line_90_deg[x].remove(&y);
        self.line_45_deg[x - y + self.n].remove(&x);
        self.line_135_deg[x + y].remove(&x);
        for i in 0..4 {
            let (x1, y1) = rect_points[i];
            let (x2, y2) = rect_points[(i+1)%4];
            if x1-y1+self.n == x2-y2+self.n {
                self.is_drawn_45_deg_edges[x1-y1+self.n].update(usize::min(x1, x2), usize::max(x1, x2), -1)
            } else {
                self.is_drawn_135_deg_edges[x1+y1].update(usize::min(x1, x2), usize::max(x1, x2), -1)
            }
        }
    }

    fn can_draw_0_deg_rotated_rect(&mut self, rect_points: &[(usize, usize); 4]) -> bool {
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
    
    fn can_draw_45_deg_rotated_rect(&mut self, rect_points: &[(usize, usize); 4]) -> bool {
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

    fn calc_score(&self) -> i32 {
        let mut sum_weight = 0;
        for x in 0..self.n {
            for y in 0..self.n {
                if self.is_drawn_points[x][y] {
                    sum_weight += ((x - (self.n-1)/2).pow(2) + (y - (self.n-1)/2).pow(2) + 1) as i32;
                }
            }
        }
        (self.coeff_score * (sum_weight as f32)).round() as i32
    }

    fn calc_weight(&self, rect_points: &[(usize, usize); 4]) -> i32 {
        let mut sum_edges = 0;
        for i in 0..4 {
            sum_edges += ((rect_points[i].0 - rect_points[(i+1)%4].0).pow(2) + (rect_points[i].1 - rect_points[(i+1)%4].1).pow(2)) as i32
        }
        let (x, y) = rect_points[0];
        (x as i32 - (self.n as i32)/2).pow(2) + (y as i32 - (self.n as i32)/2).pow(2) + 1 - (sum_edges as f32 * self.ratio).round() as i32
    }

    fn change_ratio(&mut self, new_ratio: f32) {
        self.ratio = new_ratio;
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
        m,
        ratio: (m as f32) / (n as f32),
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
    let mut detected_rects = HashSet::new();
    for x in 0..grid.n {
        for y in 0..grid.n {
            if grid.is_drawn_points[x][y] {
                detect_rects(&mut grid, &mut detected_rects, x, y);
            }
        }
    }
    let start_time = std::time::Instant::now();
    let mut best_score = 0;
    let mut best_solution = vec![];
    let ratios = [
        (grid.m as f32) / (grid.n as f32), 
        0.0, 
        (grid.m as f32 - grid.n as f32) / (grid.n as f32), 
        (grid.m as f32) / (grid.n as f32) / 2.0, 
        (grid.m as f32) / (grid.n as f32) / 4.0, 
		0.5,
      	-0.5,
        -1.0
    ];
    for &beam_width in &[2, 4, 6, 8, 10] {
        for &ratio in &ratios {
            grid.change_ratio(ratio);
            for depth in 3..8 {
                let (score, solution) = greedy(
                    grid.clone(), 
                    detected_rects.clone(), 
                    depth, 
                    beam_width,
                    start_time,
                    4.93
                );
                //eprintln!("{:.3} {} {} {}", ratio, depth, score, beam_width);
                if score > best_score {
                    best_score = score;
                    best_solution = solution;
                }
            }
        }
    }

    eprintln!("BEST_SCORE: {}", best_score);
    print_solution(best_solution);
}

fn greedy(
    mut grid: Grid, 
    mut detected_rects: HashSet<[(usize, usize); 4]>,
    depth: i32,
    beam_width: usize,
    start_time: std::time::Instant,
    duration: f32
) -> (i32, Vec<[(usize, usize); 4]>) {
    let mut best_solution = vec![];
    let mut weighted_rects = BinaryHeap::new();
    for &rect_points in detected_rects.iter() {
        if (std::time::Instant::now() - start_time).as_secs_f32() > duration {
            break;
        }
        weighted_rects.push((calc_weight(&mut grid, &rect_points, depth, beam_width), rect_points));
    }
    detected_rects.clear();

    while !weighted_rects.is_empty() {
        if (std::time::Instant::now() - start_time).as_secs_f32() > duration {
            break;
        }
        if let Some((weight, rect_points)) = weighted_rects.pop()  {
            if grid.can_draw_rect(&rect_points) {
                let now_weight = calc_weight(&mut grid, &rect_points, depth, beam_width);
                if weight == now_weight {
                    best_solution.push(rect_points);
                    grid.draw_rect(&rect_points);
                    detect_rects(&mut grid, &mut detected_rects, rect_points[0].0, rect_points[0].1);
                    for next_rect_points in detected_rects.iter() {
                        weighted_rects.push((calc_weight(&mut grid, next_rect_points, depth, beam_width), *next_rect_points));
                    }
                    detected_rects.clear();
                } else {
                    weighted_rects.push((now_weight, rect_points));
                }
            }
        }
    }
    let score = grid.calc_score();
    (score, best_solution)
}


fn calc_weight(grid: &mut Grid, rect_points: &[(usize, usize); 4], depth: i32, beam_width: usize) -> i32 {
    if depth == 0 {
        return 0;
    }
    let now_weight = grid.calc_weight(rect_points);
    let mut max_weight = now_weight;
    grid.draw_rect(&rect_points);
    let mut next_rects = HashSet::new();
    let mut weighted_rects = BinaryHeap::new();
    detect_rects(grid, &mut next_rects, rect_points[0].0, rect_points[0].1);
    for next_rect in next_rects {
        weighted_rects.push((grid.calc_weight(&next_rect), next_rect))
    }
    for _ in 0..beam_width {
        if let Some((_, next_rect)) = weighted_rects.pop() {
            max_weight = i32::max(max_weight, 
                now_weight + calc_weight(grid, &next_rect, depth-1, beam_width) * 90 / 100);
        }
    }
    grid.delete_rect(&rect_points);
    return max_weight;
}

fn print_solution(solution: Vec<[(usize, usize); 4]>) {
    println!("{}", solution.len());
    for rect in solution {
        for i in 0..3{
            print!("{} {} ", rect[i].0, rect[i].1);
        }
        println!("{} {}", rect[3].0, rect[3].1);
    }
}

fn detect_rects(
    grid: &mut Grid,
    detected_rects: &mut HashSet<[(usize, usize); 4]>,
    x: usize,
    y: usize,
) {
    detect_0_deg_rotated_rect(grid, detected_rects, x, y);
    detect_45_deg_rotated_rect(grid, detected_rects, x, y);
}

fn detect_0_deg_rotated_rect(
    grid: &mut Grid,
    detected_rects: &mut HashSet<[(usize, usize); 4]>,
    x: usize,
    y: usize,
) {
    //left -> (upper or lower)
    if let Some(&left) = grid.line_0_deg[y].range(..x).next_back() {
        if let Some(&upper) = grid.line_90_deg[left].range((y + 1)..).next() {
            let rect_points = [(x, upper), (left, upper), (left, y), (x, y)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
        if let Some(&lower) = grid.line_90_deg[left].range(..y).next_back() {
            let rect_points = [(x, lower), (x, y), (left, y), (left, lower)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }

    //right -> (upper or lower)
    if let Some(&right) = grid.line_0_deg[y].range((x + 1)..).next() {
        if let Some(&upper) = grid.line_90_deg[right].range((y + 1)..).next() {
            let rect_points = [(x, upper), (x, y), (right, y), (right, upper)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
        if let Some(&lower) = grid.line_90_deg[right].range(..y).next_back() {
            let rect_points = [(x, lower), (right, lower), (right, y), (x, y)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }

    //lower -> (right or left)
    if let Some(&lower) = grid.line_90_deg[x].range(..y).next_back() {
        if let Some(&right) = grid.line_0_deg[lower].range((x + 1)..).next() {
            let rect_points = [(right, y), (x, y), (x, lower), (right, lower)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・ - ・
        //|
        //・
        if let Some(&right) = grid.line_0_deg[y].range((x + 1)..).next() {
            let rect_points = [(right, lower), (right, y), (x, y), (x, lower)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }


        if let Some(&left) = grid.line_0_deg[lower].range(..x).next_back() {
            let rect_points = [(left, y), (left, lower), (x, lower), (x, y)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・ - ・
        //     |
        //     ・
        if let Some(&left) = grid.line_0_deg[y].range(..x).next_back() {
            let rect_points = [(left, lower), (x, lower), (x, y), (left, y)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }
    

    //upper -> (right or left)
    if let Some(&upper) = grid.line_90_deg[x].range((y + 1)..).next() {
        if let Some(&right) = grid.line_0_deg[upper].range((x + 1)..).next() {
            let rect_points = [(right, y), (right, upper), (x, upper), (x, y)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・
        //|
        //・ - ・
        if let Some(&right) = grid.line_0_deg[y].range((x + 1)..).next() {
            let rect_points = [(right, upper), (x, upper), (x, y), (right, y)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        if let Some(&left) = grid.line_0_deg[upper].range(..x).next_back() {
            let rect_points = [(left, y), (x, y), (x, upper), (left, upper)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //     ・
        //     |
        //・ - ・
        if let Some(&left) = grid.line_0_deg[y].range(..x).next_back() {
            let rect_points = [(left, upper), (left, y), (x, y), (x, upper)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
        
    }
}

fn detect_45_deg_rotated_rect(
    grid: &mut Grid,
    detected_rects: &mut HashSet<[(usize, usize); 4]>,
    x: usize,
    y: usize,
) {
    let n = grid.n;
    //left -> (upper or lower)
    if let Some(&left) = grid.line_45_deg[x - y + n].range(..x).next_back() {
        if let Some(&lower) = grid.line_135_deg[2*left+y-x].range((left + 1)..).next() {
            let rect_points = [
                (x+lower-left, y+left-lower), 
                (x, y), 
                (left, y+left-x), 
                (lower, y+2*left-lower-x)
            ];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
        if let Some(&upper) = grid.line_135_deg[2*left+y-x].range(..left).next_back() {
            let rect_points = [
                (x+upper-left, y-upper+left), 
                (upper, y+2*left-upper-x), 
                (left, y+left-x), 
                (x, y)
            ];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }

    //right -> (upper or lower)
    if let Some(&right) = grid.line_45_deg[x - y + n].range((x + 1)..).next() {
        if let Some(&lower) = grid.line_135_deg[2*right+y-x].range((right + 1)..).next() {
            let rect_points = [
                (x+lower-right, y-lower+right), 
                (lower, y+2*right-lower-x),
                (right, y+right-x),
                (x, y)
            ];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            } 
        }
        if let Some(&upper) = grid.line_135_deg[2*right+y-x].range(..right).next_back() {
            let rect_points = [
                (x+upper-right, y-upper+right), 
                (x, y),
                (right, y+right-x),
                (upper, y+2*right-upper-x)
            ];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }

    //lower -> (right or left)
    if let Some(&lower) = grid.line_135_deg[x+y].range((x+1)..).next() {
        if let Some(&left) = grid.line_45_deg[2*lower-y-x+n].range(..lower).next_back() {
            let rect_points = [
                (x-lower+left, y-lower+left), 
                (left, y+x+left-2*lower), 
                (lower, y+x-lower), 
                (x, y)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・ - ・
        //     |
        //     ・
        if let Some(&left) = grid.line_45_deg[n+x-y].range(..x).next_back() {
            let rect_points = [
                (left+lower-x, left-lower+y), 
                (lower, y+x-lower), 
                (x, y), 
                (left, y-x+left)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            } 
        }


        if let Some(&right) = grid.line_45_deg[2*lower-y-x+n].range((lower+1)..).next() {
            let rect_points = [
                (x+right-lower, y+right-lower), 
                (x, y), 
                (lower, y+x-lower), 
                (right, y+x+right-2*lower)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・ - ・
        //|
        //・
        if let Some(&right) = grid.line_45_deg[n+x-y].range((x+1)..).next() {
            let rect_points = [
                (right+lower-x, right-lower+y), 
                (right, y+right-x), 
                (x, y), 
                (lower, y+x-lower)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }
    

    //upper -> (right or left)
    if let Some(&upper) = grid.line_135_deg[x+y].range(..x).next_back() {
        if let Some(&left) = grid.line_45_deg[n+2*upper-y-x].range(..upper).next_back() {
            let rect_points = [
                (x-upper+left, y-upper+left), 
                (x, y), 
                (upper, x+y-upper), 
                (left, x+y-2*upper+left)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //     ・
        //     |
        //・ - ・
        if let Some(&left) = grid.line_45_deg[x-y+n].range(..x).next_back() {
            let rect_points = [
                (upper+left-x, left-upper+y), 
                (left, y-x+left), 
                (x, y), 
                (upper, y+x-upper)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        if let Some(&right) = grid.line_45_deg[n+2*upper-y-x].range((upper+1)..).next() {
            let rect_points = [
                (x+right-upper, y+right-upper), 
                (right, y+x-2*upper+right),
                (upper, y-upper+x), 
                (x, y)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }

        //・
        //|
        //・ - ・
        if let Some(&right) = grid.line_45_deg[x-y+n].range((x+1)..).next() {
            let rect_points = [
                (upper+right-x, right-upper+y), 
                (upper, y+x-upper), 
                (x, y), 
                (right, y+upper-x)];
            if grid.can_draw_rect(&rect_points) {
                detected_rects.insert(rect_points);
            }
        }
    }
}
//
