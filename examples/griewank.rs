use ag::*;
use rand::Rng;

const SIZE: usize = 100;
const MINV: f64 = -10.0;
const MAXV: f64 = 10.0;

fn scale(a: f64) -> f64 {
    if a > MAXV {MINV + (a - MAXV)}
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
        EPop {v: t}
    }
    fn eval<U:UserData<EPop>>(&mut self,_u:&U) -> f64 {
        let t = &self.v;
	let (mut sum,mut prod) = (0.,1.);
	for (i,o) in t.iter().enumerate().take(SIZE) {
            sum += o * o;
	    prod = prod * o.cos()/((i+1) as f64).sqrt();
	}
	let res = 10.-(sum/4000.-prod);
        res.max(0.)
    }
    fn dist(&self, u: &Self) -> f64 {
	let (t1,t2,mut d) = (&self.v,&u.v,0.);
	for i in 0..SIZE {d += (t1[i]-t2[i])*(t1[i]-t2[i])}
        d.sqrt()
    }
    fn mutate(&self, r: &mut Trng) -> EPop {
	let mut t = self.v.clone();
	let i = r.gen_range(0..SIZE);
        t[i] = scale(t[i] + r.gen_range(-0.5..0.5));
        EPop {v: t}
    }
    fn cross(e1: &mut EPop,e2: &mut EPop,r: &mut Trng) {
        let a: f64 = r.gen_range(-0.5..1.5);
	let i = r.gen_range(0..SIZE);
        let b1 = a * e1.v[i] + (1.0 - a) * e2.v[i];
        let b2 = a * e2.v[i] + (1.0 - a) * e1.v[i];
        e1.v[i] = scale(b1);
        e2.v[i] = scale(b2);
