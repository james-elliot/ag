use ag::*;
use rand::Rng;

// Find the polynomial function of degree SIZE-1 that approximates best cos(x) on [XMIN,XMAX]
// Criteria: min(I=integrate((cos(x)-a-b*x-c*x^2...)^2))
// Integral is computed by Taylor sum with NB_STEPS
// For SIZE=3, XMIN=0 and XMAX=1, the optimal COMPUTED solution by Mathematica using integration is:
// a=1.00341, b=-0.0365365, c=-0.43101 and I = 2.24736.10^(-6) (1/I=444967)
const SIZE: usize = 3;
const MINV: f64 = -10.0;
const MAXV: f64 = 10.0;
const XMIN:f64 = 0.;
const XMAX:f64 = 1.;
const NB_STEPS: usize = 100;

fn scale(a: f64) -> f64 {
    if a > MAXV {MINV + (a - MAXV)}
    else if a < MINV {MAXV + (a - MINV)}
    else {a}
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
        EPop {v: t}
    }

    fn eval<U:UserData<EPop>>(&mut self,_u:&U) -> f64 {
        let t = &self.v;
	let mut sum = 0.;
	const STEP:f64 = (XMAX-XMIN) / (NB_STEPS as f64);
	for i in 0..NB_STEPS {
	    let x = XMIN+(i as f64)*STEP;
	    let mut tsum = 0.;
	    let mut xp = 1.;
	    for v in t.iter().take(SIZE) {
		tsum += v * xp;
		xp *= x;
	    }
	    sum += (tsum-x.cos())*(tsum-x.cos());
	}
	sum /= NB_STEPS as f64;
        1./sum
    }

    fn dist(&self, u: &Self) -> f64 {
	let (t1,t2,mut d) = (&self.v,&u.v,0.);
	for i in 0..SIZE {d += (t1[i]-t2[i])*(t1[i]-t2[i])}
        d.sqrt()
    }

    fn mutate(&self, r: &mut Trng) -> EPop {
	let mut t = self.v.clone();
	/*
	let i = r.gen_range(0..SIZE);
        t[i] = scale(t[i] * (1.+r.gen_range(-0.05..0.05)));
	*/
	for v in t.iter_mut().take(SIZE) {
            *v = scale(*v * (1.+r.gen_range(-0.05..0.05)));
	}
        EPop {v: t}
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
        EPop {v:t}
    }
}

fn main() {
    let mut u = UData {};
    let (bests,ptime,wtime) = ag::<EPop,UData>(None,&mut u);
    println!("Bests: {:?}\nprocess_time: {:?}\nwall_clock_time: {:?}", bests,ptime,wtime);
}
