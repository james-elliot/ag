use rand::{Rng,SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use kodama::{Method, linkage};
use serde::{Deserialize, Serialize};
use json_comments::StripComments;
use std::{fs,env};
use std::path::Path;
use std::time::{Instant,Duration};
use cpu_time::ProcessTime;


pub type Trng=rand_chacha::ChaCha8Rng;
pub trait UserData<T:ElemPop>: Send +Sync {
    fn update(&mut self,p: &Pop<T>);
}
pub trait ElemPop: Clone + Send +Sync + std::fmt::Debug {
    fn new(r: &mut Trng) -> Self;
    fn eval<U:UserData<Self>>(&self,u:&U) -> f64;
    fn dist(&self, u: &Self) -> f64;
    fn mutate(&self, r: &mut Trng) -> Self;
    fn cross(e1: &mut Self,e2: &mut Self,r: &mut Trng);
    fn barycenter(e1: &Self, e2: &Self, n1: u32, n2: u32) -> Self;
}

#[derive(Debug, Clone)]
pub struct Chromosome<T: ElemPop> {
    pub data: Arc<T>,
    pub r_fit: Option<f64>,
    s_fit: f64,
}
pub type Pop<T> = Vec<Chromosome<T>>;

fn normalize_hard<T: ElemPop>(mut p: Pop<T>) -> Pop<T> {
    let (mini, maxi) = p
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mini, maxi), x| {
            (mini.min(x.s_fit), maxi.max(x.s_fit))
        });
    let diff = maxi - mini;
    for v in p.iter_mut() {
        if diff == 0. {v.s_fit = 1.0}
	else {v.s_fit = (v.s_fit - mini) / diff}
    }
    return p;
}

fn normalize_simple<T: ElemPop>(mut p: Pop<T>) -> Pop<T> {
    let maxi = p.iter().fold(f64::NEG_INFINITY, |acc, x| acc.max(x.s_fit));
    if maxi<=0. {panic!("All fitnesses <=0 in normalize_simple")}
    for v in p.iter_mut() {v.s_fit = (v.s_fit / maxi).max(0.)}
    return p;
}

fn normalize_none<T: ElemPop>(mut p: Pop<T>) -> Pop<T> {
    let maxi = p.iter().fold(f64::NEG_INFINITY, |acc, x| acc.max(x.s_fit));
    if maxi<=0. {panic!("All fitnesses <=0 in normalize_none")}
    for v in p.iter_mut() {v.s_fit = v.s_fit.max(0.)}
    return p;
}

fn sigma_truncation<T: ElemPop>(mut p: Pop<T>) -> Pop<T> {
    let nb_elems = p.len() as f64;
    let (sum, ssum) = p.iter().fold((0., 0.), |(a1, a2), x| {(a1 + x.s_fit, a2 + x.s_fit * x.s_fit)});
    let mean = sum / nb_elems;
    let sigma = (ssum / nb_elems - mean * mean).sqrt();
    for x in p.iter_mut() {x.s_fit = (0.0_f64).max(x.s_fit - (mean - 2.0 * sigma))}
    return p;
}

fn ranking<T: ElemPop>(mut p: Pop<T>) -> Pop<T> {
    let nb_elems = p.len();
    p.sort_unstable_by(|a, b| b.s_fit.partial_cmp(&a.s_fit).unwrap());
    for (i, x) in p.iter_mut().enumerate() {x.s_fit = (nb_elems - i) as f64}
    return p;
}

fn scale_fitness<T: ElemPop>(mut p: Pop<T>,scaling: u64,normalize: u64) -> Pop<T> {
    for x in p.iter_mut() {x.s_fit = x.r_fit.unwrap();}
    match scaling {
	0 => {},
	1 => p = sigma_truncation(p),
	2 => p = ranking(p),
	_ => panic!("SCALING invalid")
    }
    match normalize {
	0 => p=normalize_none(p),
	1 => p = normalize_simple(p),
	2 => p = normalize_hard(p),
	_ => panic!("NORMALIZE invalid")
    }
    return p;
}

fn eval_pop<T: ElemPop, U:UserData<T>>(mut p: Pop<T>,u:&U, par: bool, evolutive: bool) -> Pop<T> {
    if par {
	p.par_iter_mut().for_each(|v| {
	    if v.r_fit == None || evolutive {v.r_fit = Some(v.data.eval(u))}
	})
    }
    else {
	p.iter_mut().for_each(|v| {
	    if v.r_fit == None || evolutive {v.r_fit = Some(v.data.eval(u))}
	})
    }
    return p;
}

fn new_pop<T: ElemPop>(nb_elems: usize,rng: &mut Trng) -> Pop<T> {
    let mut p: Pop<T> = Vec::with_capacity(nb_elems);
    for _i in 0..nb_elems {p.push(Chromosome {data: Arc::new(ElemPop::new(rng)), r_fit: None, s_fit: 0.,})}
    return p;
}

fn find_best<T: ElemPop>(p: &Pop<T>) -> usize {
    let (mut b,mut ib) = (f64::NEG_INFINITY,None);
    for (ind, e) in p.iter().enumerate() {
	let v = e.r_fit.unwrap();
        if v > b {b = v;ib = Some(ind)}
    }
    return ib.unwrap();
}

// Reproduce with stochastic remainders
fn reproduce_pop<T: ElemPop>(mut oldp: Pop<T>, bests: &Vec<usize>, rng: &mut Trng) -> Pop<T> {
    let nb_elems = oldp.len();
    let mut p: Pop<T> = Vec::with_capacity(nb_elems);
    let mut rmt: Vec<f64> = Vec::with_capacity(nb_elems);

    let ssum1 = oldp.iter().fold(0., |acc, x| acc + x.s_fit);
    for e in oldp.iter_mut() {e.s_fit = e.s_fit / ssum1 * (nb_elems as f64)+1e-10}
    let mut t_rem = nb_elems as i64;
    for ind in bests.iter() {
        p.push(oldp[*ind].clone());
        oldp[*ind].s_fit = (0.0_f64).max(oldp[*ind].s_fit - 1.);
        t_rem = t_rem - 1;
    }
    let ssum2 = oldp.iter().fold(0., |acc, x| acc + x.s_fit);
    let mut sum = 0;
    let mut fsum = 0.0;
    for v in oldp.iter() {
        let nb = (t_rem as f64) * v.s_fit / ssum2;
        let inb = nb as i64;
        let rem = nb - (inb as f64);
        sum = sum + inb;
        fsum = fsum + rem;
        rmt.push(rem);
        for _i in 0..inb {p.push(v.clone())}
    }
    for v in rmt.iter_mut() {*v = *v / fsum}
    let mut prev = 0.;
    for v in rmt.iter_mut() {*v = *v + prev;prev = *v}
    let rem = t_rem - sum;
    for _i in 0..rem {
	let f = rng.gen_range(0.0..1.0);
	let (mut up,mut low)=(nb_elems-1,0);
	while up!=low {
	    let med = (up+low)/2;
	    if f<=rmt[med] {up=med} else {low=med+1}
	}
	p.push(oldp[up].clone());
    }
    return p;
}

fn cross_mut<T: ElemPop>(mut oldp: Pop<T>, nb_prot: usize, pcross: f64, pmut: f64, rng: &mut Trng) -> Pop<T> {
    let nb_elems = oldp.len();
    let mut p: Pop<T> = Vec::with_capacity(nb_elems);
    let nb_cross = ((nb_elems as f64) * pcross / 2.0) as usize;
    let nb_mut = ((nb_elems as f64) * pmut) as usize;
    let nbr = (nb_elems as usize) - nb_prot - nb_mut - nb_cross * 2;
    for i in 0..nb_prot {p.push(oldp[i].clone())}
    for _i in 0..nb_mut {
        let ind = rng.gen_range(0..nb_elems);
        let d = oldp[ind].data.mutate(rng);
        p.push(Chromosome {data: Arc::new(d),r_fit: None,s_fit: 0.,});
    }
    for _i in 0..nb_cross {
        let ind1 = rng.gen_range(0..nb_elems);
        let ind2 = (ind1 + rng.gen_range(0..nb_elems - 1) + 1) % nb_elems;
        let mut a = Arc::make_mut(&mut oldp[ind1].data).clone();
        let mut b = Arc::make_mut(&mut oldp[ind2].data).clone();
        ElemPop::cross(&mut a, &mut b, rng);
        p.push(Chromosome {data: Arc::new(a), r_fit: None, s_fit: 0.,});
        p.push(Chromosome {data: Arc::new(b), r_fit: None, s_fit: 0.,});
    }
    for _i in 0..nbr {
        let ind = rng.gen_range(0..nb_elems);
        p.push(oldp[ind].clone());
    }
    return p;
}

#[derive(Debug)]
struct Cluster<T: ElemPop> {
    center: T,
    elems: Vec<usize>,
    nb_elems: u32,
    best: usize,
    v_best: f64,
}

fn dendro_clustering<T: ElemPop>(p: &Pop<T>, dmax: f64, pmax: f64) -> Vec<Cluster<T>> {
    let mut clus: Vec<Cluster<T>> = Vec::with_capacity(p.len());
    let mut condensed = vec![];
    for row in 0..p.len() - 1 {
        for col in row + 1..p.len() {
            condensed.push(p[row].data.dist(&p[col].data));
        }
    }
    for i in 0..p.len() {
	let mut c = Cluster {
	    center: Arc::make_mut(&mut p[i].data.clone()).clone(),
            elems: Vec::new(),
            nb_elems: 1,
            best: i,
            v_best: p[i].r_fit.unwrap(),
        };
        c.elems.push(i);
	clus.push(c);
    }
    let dend = linkage(&mut condensed, p.len(), Method::Average);
    let steps = dend.steps();
    let nb_clus_max = ((p.len() as f64) * pmax+0.5) as usize;
    let mut nb_clus = p.len();
    for s in steps.iter() {
	if nb_clus <= nb_clus_max {break}
	if s.dissimilarity > dmax {break}
	let (i,j) = (s.cluster1,s.cluster2);
	let mut k = i;
	if clus[j].v_best > clus[i].v_best {k=j};
	let mut c = Cluster {
            center: ElemPop::barycenter(
		&clus[i].center,&clus[j].center,
		clus[i].nb_elems, clus[j].nb_elems),
            elems: clus[i].elems.clone(),
            nb_elems: clus[i].nb_elems+clus[j].nb_elems,
            best: clus[k].best,
            v_best: clus[k].v_best,
        };
	c.elems.append(&mut clus[j].elems);
	clus.push(c);
	clus[i].nb_elems=0;
	clus[j].nb_elems=0;
	nb_clus = nb_clus-1;
    }
    clus.retain(|c| c.nb_elems > 0);
    return clus;
}

fn dyn_clustering<T: ElemPop>(p: &Pop<T>, dmax: f64) -> Vec<Cluster<T>> {
    let nb_elems = p.len();
    let mut clus: Vec<Cluster<T>> = Vec::with_capacity(nb_elems);
    for (ind, e) in p.iter().enumerate() {
        let (mut bd,mut bi) = (f64::INFINITY,None);
        for (ic, c) in clus.iter().enumerate() {
            let dist = e.data.dist(&c.center);
            if dist < bd {bd = dist;bi = Some(ic)}
        }
        if bd > dmax {
            let mut c = Cluster {
		center: Arc::make_mut(&mut e.data.clone()).clone(),
                elems: Vec::new(),
                nb_elems: 1,
                best: ind,
                v_best: e.r_fit.unwrap(),
            };
            c.elems.push(ind);
            clus.push(c);
        }
	else {
            let bi = bi.unwrap();
            clus[bi].center = ElemPop::barycenter(
		&clus[bi].center, &e.data,
		clus[bi].nb_elems, 1);
            clus[bi].nb_elems = clus[bi].nb_elems + 1;
            clus[bi].elems.push(ind);
            if e.r_fit.unwrap() > clus[bi].v_best {
                clus[bi].v_best = e.r_fit.unwrap();
                clus[bi].best = ind;
            }
        }
    }
    return clus;
}

fn share_fitness<T: ElemPop>(mut p: Pop<T>, clus: &Vec<Cluster<T>>, spenalty : u64) -> Pop<T> {
    let nb = p.len() as f64;
    for c in clus.iter() {
	let k = c.nb_elems as f64;
        for i in c.elems.iter() {
	    match spenalty {
		0 => {}
		1 => p[*i].s_fit = p[*i].s_fit / k,
		2 => p[*i].s_fit = p[*i].s_fit * (1.-((k-1.)/nb)),
		_ => panic!("Invalid spenalty")
	    }
	}
    }
    return p;
}

fn get_bests<T: ElemPop>(mut clus: Vec<Cluster<T>>, mut nbest:Vec<usize>,sfactor: f64, mut max_best: u64) -> Vec<usize> {
    nbest.clear();
    clus.sort_unstable_by(|a, b| b.v_best.partial_cmp(&a.v_best).unwrap());
    let vbest = clus[0].v_best;
    for c in clus.iter() {
        if c.v_best >= vbest * sfactor {
            nbest.push(c.best);
            max_best = max_best - 1;
            if max_best == 0 {break;}
        }
	else {break;}
    }
    return nbest;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Params {
    pub nb_elems: usize,
    pub nb_gen: u64,
    pub pmut: f64,
    pub pcross: f64,
    pub evolutive: bool,
    pub scaling: u64,
    pub normalize: u64,
    pub elitist: bool,
    pub sharing: u64,
    pub sfactor: f64,
    pub dmax: f64,
    pub pmax: f64,
    pub spenalty : u64,
    pub parallel: bool,
    pub verbose:u64,
    pub seed: u64,
    pub ctrlc: bool
}

#[derive(Default,Debug)]
pub struct Timing {
    pub eval:Duration,
    pub scaling:Duration,
    pub clustering:Duration,
    pub sshare:Duration,
    pub crossmut: Duration,
    pub reproduce: Duration,
    pub total:Duration
}

pub fn ag<T:ElemPop,U:UserData<T>>(param:Option<Params>,u:&mut U)-> (Vec<(T,f64)>,Timing,Timing) {
    let args: Vec<String> = env::args().collect();
    let path = Path::new(&args[0]);
    let name = path.file_name().unwrap();
    let par:Params;

    match param {
	None => {
	    let paths = [".","examples"];
	    let mut i =0;
	    let path =
		loop {
		    let mut new_name = Path::new(paths[i]).join(name);
		    new_name = new_name.with_extension("json");
		    if fs::metadata(new_name.clone()).is_ok() {break new_name};
		    i=i+1;
		    if i==paths.len() {panic!("No parameters file found")}
		};
	    println!("Parameter file {:?} found",path);
	    let contents = fs::read_to_string(path)
		.expect("Something went wrong reading the file");
	    let stripped = StripComments::new(contents.as_bytes());
	    par = serde_json::from_reader(stripped).unwrap();
	}
	Some (p) => {par=p}
    }
    if par.verbose>=1 {println!("{:?}",par)}

    let shared = Arc::new(AtomicBool::new(false));
    let shared1 = Arc::clone(&shared);
    if par.ctrlc {
	ctrlc::set_handler(move || {
	    shared1.store(true,Ordering::Relaxed);
            println!("received Ctrl+C, exiting now");
	})
	    .expect("Error setting Ctrl-C handler")
    }

    if par.parallel {
	let nb = num_cpus::get_physical();
	rayon::ThreadPoolBuilder::new().num_threads(nb).build_global().unwrap();
	if par.verbose>=1 {println!("Using {} cpus",nb)}
    }
    
    let mut rng = Trng::seed_from_u64(par.seed);
    let mut p: Pop<T> = new_pop(par.nb_elems,&mut rng);
    if par.verbose>=3 {for val in p.iter() {println!("Init: {:?}", val)}}

    let mut bests=Vec::new();
    let (mut tpi,mut twi):(Timing,Timing)=(Default::default(),Default::default());
    let (sptt,swtt) = (ProcessTime::now(),Instant::now());
    let mut num = 0;
    loop {
	bests.clear();
	let (spt,swt) = (ProcessTime::now(),Instant::now());
        p = eval_pop(p,u,par.parallel,par.evolutive);
	tpi.eval=tpi.eval.saturating_add(spt.elapsed());
	twi.eval=twi.eval.saturating_add(swt.elapsed());
	let ibest = find_best(&p);
	if par.verbose>=2 {println!("Gen: {:?} Best: {:?}", num, p[ibest])}
	if par.elitist {bests.push(ibest)}
	
	let (spt,swt) = (ProcessTime::now(),Instant::now());
        p = scale_fitness(p,par.scaling,par.normalize);
	tpi.scaling=tpi.scaling.saturating_add(spt.elapsed());
	twi.scaling=twi.scaling.saturating_add(swt.elapsed());
        if par.verbose>=3 {for val in p.iter() {println!("Scale: {:?}", val)}}

	if par.sharing > 0 {
	    let (spt,swt) = (ProcessTime::now(),Instant::now());
            let clusters;
	    match par.sharing {
		1 => clusters = dyn_clustering(&p, par.dmax),
		2 => clusters = dendro_clustering(&p, par.dmax, par.pmax),
		_ => panic!("Sharing is 0, 1 or 2")
	    }
            if par.verbose>=3 {for c in clusters.iter() {println!("{:?}", c)}}
            p = share_fitness(p, &clusters, par.spenalty);
            if par.verbose>=3 {for val in p.iter() {println!("Share: {:?}", val)}}
	    tpi.clustering=tpi.clustering.saturating_add(spt.elapsed());
	    twi.clustering=twi.clustering.saturating_add(swt.elapsed());
	    if par.verbose>=2 {println!("nb_clus={:?}",clusters.len())}
	    if par.sfactor<=1.0 {
		let (spt,swt) = (ProcessTime::now(),Instant::now());
		let max_best = ((par.nb_elems as f64) * (1.0 - par.pmut - par.pcross)) as u64;
		bests = get_bests(clusters, bests,par.sfactor, max_best);
		tpi.sshare=tpi.sshare.saturating_add(spt.elapsed());
		twi.sshare=twi.sshare.saturating_add(swt.elapsed());
	    }
	    if par.verbose>=2 {println!("nb_clus_saved={:?}",bests.len())}
	}
        if par.verbose>=3 {for b in bests.iter() {println!("best: {:?}", b)}}

	if shared.load(Ordering::Relaxed) || num==par.nb_gen {
	    if bests.is_empty() {bests.push(ibest)}
	    break
	}
	num=num+1;

	u.update(&p);
	
	let (spt,swt) = (ProcessTime::now(),Instant::now());
        p = reproduce_pop(p, &bests,&mut rng);
	tpi.reproduce=tpi.reproduce.saturating_add(spt.elapsed());
	twi.reproduce=twi.reproduce.saturating_add(swt.elapsed());
        if par.verbose>=3 {for val in p.iter() {println!("Reproduce: {:?}", val)}}

	let (spt,swt) = (ProcessTime::now(),Instant::now());
	p = cross_mut(p, bests.len(), par.pcross, par.pmut, &mut rng);
	tpi.crossmut=tpi.crossmut.saturating_add(spt.elapsed());
	twi.crossmut=twi.crossmut.saturating_add(swt.elapsed());
        if par.verbose>=3 {for val in p.iter() {println!("CrossMut: {:?}", val)}}
    }
    
    let mut res = Vec::new();
    for i in bests.iter() {
	res.push((Arc::make_mut(&mut p[*i].data.clone()).clone(),p[*i].r_fit.unwrap()));
    }
    res.sort_unstable_by(|(_,v1), (_,v2)| v1.partial_cmp(v2).unwrap());
    tpi.total=tpi.total.saturating_add(sptt.elapsed());
    twi.total=twi.total.saturating_add(swtt.elapsed());
    return (res,tpi,twi)
}
