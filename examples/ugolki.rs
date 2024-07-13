use std::io::prelude::*;
use std::process::{Stdio,Command};

fn test(v1:Vec<i32>,v2:Vec<i32>,t:f64) -> i32 {
    let v1:Vec<String> = v1.into_iter().map(|x| x.to_string() ).collect();
    let mut v:Vec<String> = vec!["1".to_string(),t.to_string()]; 
    v.extend(v1);
    let mut process1 =
	match Command::new("/home/alliot/ISAE/JEUX/UGOLKI/PERSO/ugolki.out")
	.args(v)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn() {
            Err(why) => {
		println!("couldn't spawn P1: {}", why);
		return -1},
            Ok(process) => process,
	};

    let v2:Vec<String> = v2.into_iter().map(|x| x.to_string() ).collect();
    let mut v:Vec<String> = vec!["-1".to_string(),t.to_string()]; 
    v.extend(v2);
    let mut process2 =
	match Command::new("/home/alliot/ISAE/JEUX/UGOLKI/PERSO/ugolki.out")
	.args(v)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn() {
            Err(why) => {
		println!("couldn't spawn P2: {}", why);
		if let Err(why) = process1.kill() { println!("couldn't kill P1: {}", why) };
		return -1
	    },
            Ok(process) => process,
	};

    loop {
	let mut buffer = [0; 10];
	let n = match process1.stdout.as_mut().unwrap().read(&mut buffer[..]) {
	    Ok(n) => n,
	    Err(_) => break
	};
//	println!("Responded with:{} : {:?}\n", n,buffer);
	if n==0 {break;}
	match process2.stdin.as_mut().unwrap().write_all(&buffer[0..n]){
	    Ok(_) => (),
	    Err(_) => break
	}
	match process2.stdin.as_mut().unwrap().flush() {
	    Ok(_) => (),
	    Err(_) => break
	}

	let mut buffer = [0; 10];
	let n = match process2.stdout.as_mut().unwrap().read(&mut buffer[..]) {
	    Ok(n) => n,
	    Err(_) => break
	};
//	println!("Responded with:{} : {:?}\n", n,buffer);
	if n==0 {break;}
	match process1.stdin.as_mut().unwrap().write_all(&buffer[0..n]){
	    Ok(_) => (),
	    Err(_) => break
	}
	match process1.stdin.as_mut().unwrap().flush() {
	    Ok(_) => (),
	    Err(_) => break
	}
    }
    
    let mut res= -1;
    match process1.try_wait() {
	Ok(Some(status)) =>
	    if let Some(v) = status.code() { res=v },
	Ok(None) => {
            println!("Process1 not finished");
	    if let Err(why) = process1.kill() { println!("couldn't kill P1: {}", why) }
	}
	Err(e) => println!("error attempting to wait for process1: {e}"),
    }

    match process2.try_wait() {
	Ok(Some(status)) =>
	    if let Some(v) = status.code() { res=v },
	Ok(None) => {
            println!("Process2 not finished");
	    if let Err(why) = process2.kill() { println!("couldn't kill P2: {}", why) }
	}
	Err(e) => println!("error attempting to wait for process2: {e}"),
    }

    res
}


use ag::*;
use rand::Rng;

const SIZE: usize = 36;
const MINV: i32 = 0;
const MAXV: i32 = 40;

fn scale(a: i32) -> i32 {
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
    v: Vec<i32>,
}
impl ElemPop for EPop {
    fn new(r: &mut Trng) -> EPop {
	let mut t = Vec::with_capacity(SIZE);
	for _i in 0..SIZE {t.push(r.gen_range(MINV..MAXV))}
        EPop {v: t}
    }
    
    fn eval<U:UserData<EPop>>(&mut self,_u:&U) -> f64 {
        let t = &self.v;
	let res1 = test(t.to_vec(),vec![],0.1);
	let res2 = test(vec![],t.to_vec(),0.1);
	let res=res1-res2+32;
	res as f64
    }
    
    fn dist(&self, u: &Self) -> f64 {
	let (t1,t2,mut d) = (&self.v,&u.v,0.);
	for i in 0..SIZE {d += ((t1[i]-t2[i])*(t1[i]-t2[i])) as f64}
        d.sqrt()
    }
    
    fn mutate(&self, r: &mut Trng) -> EPop {
	let mut t = self.v.clone();
	let i = r.gen_range(0..SIZE);
	let mut v = 0;
	while v==0 {
	    v=r.gen_range(-3..4);
	}
        t[i] = scale(t[i] + v);
        EPop {v: t}
    }
    
    fn cross(e1: &mut EPop,e2: &mut EPop,r: &mut Trng) {
        let a: f64 = r.gen_range(-0.5..1.5);
	let i = r.gen_range(0..SIZE);
        let b1 = a * (e1.v[i] as f64) + (1.0 - a) * (e2.v[i] as f64);
        let b2 = a * (e2.v[i] as f64) + (1.0 - a) * (e1.v[i] as f64);
        e1.v[i] = scale(b1 as i32);
        e2.v[i] = scale(b2 as i32);
    }
    
    fn barycenter(e1: &Self, e2: &Self, n1: u32, n2: u32) -> Self {
	let mut t = Vec::with_capacity(SIZE);
        let (fn1,fn2) = (n1 as f64,n2 as f64);
	for i in 0..SIZE {t.push(((fn1 * (e1.v[i] as f64)+ fn2 * (e2.v[i] as f64)) / (fn1 + fn2)) as i32)}
        EPop {v:t}
    }
}

fn main() {
    let mut u = UData {};
    let (bests,ptime,wtime) = ag::<EPop,UData>(None,&mut u);
    println!("Bests: {:?}\nprocess_time: {:?}\nwall_clock_time: {:?}", bests,ptime,wtime);
}
