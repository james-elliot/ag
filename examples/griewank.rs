use ag::*;
use rand::Rng;

const SIZE: usize = 100;
const MINV: f64 = -10.0;
const MAXV: f64 = 10.0;

fn scale(a: f64) -> f64 {
    if a > MAXV {return MINV + (a - MAXV)}
    else if a < MINV {return MAXV + (a - MINV)}
    else {return a}
}

struct UData {}
impl UserData<EPop> for UData {
    fn update(&mut self,_p: &Pop<EPop>) {
    }
}

#[derive(Debug, Clone)]
struct EPop {
    v: Vec<f64>,
}
impl ElemPop for EPop {
    fn new(r: &mut Trng) -> EPop {
	let mut t = Vec::with_capacity(SIZE);
	for _i in 0..SIZE {t.push(r.gen_range(MINV..MAXV))}
        return EPop {v: t};
    }
    fn eval<U:UserData<EPop>>(&self,_u:&U) -> f64 {
        let t = &self.v;
	let (mut sum,mut prod) = (0.,1.);
	for i in 0..SIZE {
            sum = sum + t[i] * t[i];
	    prod = prod * t[i].cos()/((i+1) as f64).sqrt();
	}
	let res = 10.-(sum/4000.-prod);
        return res.max(0.);
    }
    fn dist(&self, u: &Self) -> f64 {
	let (t1,t2,mut d) = (&self.v,&u.v,0.);
	for i in 0..SIZE {d = d+(t1[i]-t2[i])*(t1[i]-t2[i])}
        return d.sqrt();
    }
    fn mutate(&self, r: &mut Trng) -> EPop {
	let mut t = self.v.clone();
	let i = r.gen_range(0..SIZE);
        t[i] = scale(t[i] + r.gen_range(-0.5..0.5));
        return EPop {v: t}
    }
    fn cross(e1: &mut EPop,e2: &mut EPop,r: &mut Trng) {
        let a: f64 = r.gen_range(-0.5..1.5);
	let i = r.gen_range(0..SIZE);
        let b1 = a * e1.v[i] + (1.0 - a) * e2.v[i];
        let b2 = a * e2.v[i] + (1.0 - a) * e1.v[i];
        e1.v[i] = scale(b1);
        e2.v[i] = scale(b2);
    }
    fn barycenter(e1: &Self, e2: &Self, n1: u32, n2: u32) -> Self {
	let mut t = Vec::with_capacity(SIZE);
        let (fn1,fn2) = (n1 as f64,n2 as f64);
	for i in 0..SIZE {t.push((fn1 * e1.v[i] + fn2 * e2.v[i]) / (fn1 + fn2))}
        return EPop {v:t};
    }
}

fn main() {
    let mut u = UData {};
    let (e,process_time,wall_clock_time) = ag::<EPop,UData>(None,&mut u);
    println!("Bests: {:?}\nprocess_time:{:?}\nwall_clock_time:{:?}", e,process_time,wall_clock_time);
}
