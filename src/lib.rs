//! Combinators for [`rand::distr::Distribution`] objects
//!
//! This module provides tools to compose and transform probability distributions
//! in a functional style. It is designed to be generic and independent of any
//! specific data structures.

use rand::Rng;
use rand_distr::Distribution;
use rand::distr::{Iter, Uniform};
use std::marker::PhantomData;

/// A convenience to analyze the distribution of generated values
pub mod histogram;
pub use histogram::Histogram;

/// A distribution that samples from `d` but rejects values that do not satisfy predicate `p`.
#[derive(Clone)]
pub struct Filtered<T, D : Distribution<T> + Clone, P : Fn(&T) -> bool> { pub d: D, pub p: P, pub pd: PhantomData<T> }
impl <T, D : Distribution<T> + Clone, P : Fn(&T) -> bool> Distribution<T> for Filtered<T, D, P> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    loop {
      let s = self.d.sample(rng);
      if (self.p)(&s) { return s }
    }
  }
}

/// A distribution that maps values sampled from `d` using function `f`.
#[derive(Clone)]
pub struct Mapped<T, S, D : Distribution<T> + Clone, F : Fn(T) -> S + Clone> { pub d: D, pub f: F, pub pd: PhantomData<(T, S)> }
impl <T, S, D : Distribution<T> + Clone, F : Fn(T) -> S + Clone> Distribution<S> for Mapped<T, S, D, F> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> S {
    (self.f)(self.d.sample(rng))
  }
}

/// A distribution that samples from `d`, maps using `pf`, and retries if `pf` returns `None`.
#[derive(Clone)]
pub struct Collected<T, S, D : Distribution<T> + Clone, P : Fn(T) -> Option<S> + Clone> { pub d: D, pub pf: P, pub pd: PhantomData<(T, S)> }
impl <T, S, D : Distribution<T> + Clone, P : Fn(T) -> Option<S> + Clone> Distribution<S> for Collected<T, S, D, P> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> S {
    loop {
      let t = self.d.sample(rng);
      match (self.pf)(t) {
        None => {}
        Some(s) => { return s }
      }
    }
  }
}

/// A distribution that combines two independent distributions `dx` and `dy` using a function `f`.
#[derive(Clone)]
pub struct Product2<X, DX : Distribution<X> + Clone, Y, DY : Distribution<Y> + Clone, Z, F : Fn(X, Y) -> Z + Clone> { pub dx: DX, pub dy: DY, pub f: F,
  pub pd: PhantomData<(X, Y, Z)> }
impl <X, DX : Distribution<X> + Clone, Y, DY : Distribution<Y> + Clone, Z, F : Fn(X, Y) -> Z + Clone> Distribution<Z> for Product2<X, DX, Y, DY, Z, F> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Z {
    (self.f)(self.dx.sample(rng), self.dy.sample(rng))
  }
}

/// A distribution that chooses between `dx` and `dy` based on a boolean distribution `db`.
#[derive(Clone)]
pub struct Choice2<X, DX : Distribution<X> + Clone, Y, DY : Distribution<Y> + Clone, DB : Distribution<bool> + Clone> { pub dx: DX, pub dy: DY, pub db: DB,
  pub pd: PhantomData<(X, Y)> }
impl <X, DX : Distribution<X> + Clone, Y, DY : Distribution<Y> + Clone, DB : Distribution<bool> + Clone> Distribution<Result<X, Y>> for Choice2<X, DX, Y, DY, DB> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<X, Y> {
    if self.db.sample(rng) { Ok(self.dx.sample(rng)) }
    else { Err(self.dy.sample(rng)) }
  }
}

/// A distribution where the choice of the second distribution depends on the value sampled from the first.
#[derive(Clone)]
pub struct Dependent2<X, DX : Distribution<X> + Clone, Y, DY : Distribution<Y> + Clone, FDY : Fn(X) -> DY + Clone> { pub dx: DX, pub fdy: FDY,
  pub pd: PhantomData<(X, Y)> }
impl <X, DX : Distribution<X> + Clone, Y, DY : Distribution<Y> + Clone, FDY : Fn(X) -> DY + Clone> Distribution<Y> for Dependent2<X, DX, Y, DY, FDY> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Y {
    (self.fdy)(self.dx.sample(rng)).sample(rng)
  }
}

/// A stateful distribution that accumulates samples from `dx` into state `z` until `fa` returns a value.
#[derive(Clone)]
pub struct Concentrated<X, DX : Distribution<X> + Clone, A : Clone, Y, FA : Fn(&mut A, X) -> Option<Y>> { pub dx: DX, pub z: A, pub fa: FA,
  pub pd: PhantomData<(X, Y)> }
impl <X, DX : Distribution<X> + Clone, A : Clone, Y, FA : Fn(&mut A, X) -> Option<Y>> Distribution<Y> for Concentrated<X, DX, A, Y, FA> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Y {
    let mut a = self.z.clone();
    loop {
      match (self.fa)(&mut a, self.dx.sample(rng)) {
        None => {}
        Some(y) => { return y }
      }
    }
  }
}

/// A distribution that expands a single sample from `dx` into a sequence of values.
#[derive(Clone)]
pub struct Diluted<X, DX : Distribution<X> + Clone, A : Clone, Y, FA : Fn(X) -> A, FAY : Fn(&mut A) -> Option<Y>> { pub dx: DX, pub fa: FA, pub fay: FAY,
  pub pd: PhantomData<(X, A, Y)> }
impl <X, DX : Distribution<X> + Clone, A : Clone, Y, FA : Fn(X) -> A, FAY : Fn(&mut A) -> Option<Y>> Distribution<Y> for Diluted<X, DX, A, Y, FA, FAY> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Y {
    let mut a = (self.fa)(self.dx.sample(rng));
    (self.fay)(&mut a).expect("fay returns at least once per fa call")
  }

  fn sample_iter<R>(self, _rng: R) -> Iter<Self, R, Y> where R : Rng, Self : Sized {
    panic!("This function returning a concrete object makes it impossible to override the iterator behavior")
  }
}

/// A constant distribution that always returns `element`.
#[derive(Clone)]
pub struct Degenerate<T : Clone> { pub element: T }
impl <T : Clone> Distribution<T> for Degenerate<T> {
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    self.element.clone()
  }
}

/// A categorical distribution that selects from `elements` based on an index distribution `ed`.
#[derive(Clone)]
pub struct Categorical<T : Clone, ElemD : Distribution<usize> + Clone> { pub elements: Vec<T>, pub ed: ElemD }
impl <T : Clone, ElemD : Distribution<usize> + Clone> Distribution<T> for Categorical<T, ElemD> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
    self.elements[self.ed.sample(rng)].clone()
  }
}

/// Creates a categorical distribution where elements are chosen with probability proportional to their weights.
pub fn ratios<T : Clone>(ep: impl IntoIterator<Item=(T, usize)>) -> Categorical<T, Mapped<usize, usize, Uniform<usize>, impl Fn(usize) -> usize + Clone>> {
  let mut elements = vec![];
  let mut cdf = vec![];
  let mut sum = 0;
  for (e, r) in ep.into_iter() {
    elements.push(e);
    cdf.push(sum);
    sum += r;
  }
  let us = Uniform::try_from(0..sum).unwrap();
  Categorical {
    elements,
    // it's much cheaper to draw many samples at once, but the current Distribution API is broken
    ed: Mapped{ d: us, f: move |x| { match cdf.binary_search(&x) {
      Ok(i) => { i }
      Err(i) => { i - 1 }
    }}, pd: PhantomData::default() }
  }
}

/// A distribution that generates a vector of items with length sampled from `lengthd` and items from `itemd`.
#[derive(Clone)]
pub struct Repeated<T, LengthD : Distribution<usize>, ItemD : Distribution<T>> { pub lengthd: LengthD, pub itemd: ItemD, pub pd: PhantomData<T> }
impl <T, LengthD : Distribution<usize>, ItemD : Distribution<T>> Distribution<Vec<T>> for Repeated<T, LengthD, ItemD> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<T> {
    let l = self.lengthd.sample(rng);
    Vec::from_iter(std::iter::repeat_with(|| self.itemd.sample(rng)).take(l))
  }
}

/// A distribution that generates a vector of bytes by sampling from `mbd` until `None` is returned.
#[derive(Clone)]
pub struct Sentinel<MByteD : Distribution<Option<u8>> + Clone> { pub mbd: MByteD }
impl <MByteD : Distribution<Option<u8>> + Clone> Distribution<Vec<u8>> for Sentinel<MByteD> {
  fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<u8> {
    let mut v = vec![];
    while let Some(e) = self.mbd.sample(rng) {
      v.push(e)
    }
    v
  }
}

#[cfg(test)]
mod tests {
  use rand::rngs::StdRng;
  use rand::SeedableRng;
  use rand::distr::Uniform;
  use crate::*;

  #[test]
  fn monte_carlo_pi() {
    #[cfg(not(miri))]
    const SAMPLES: usize = 100000;
    #[cfg(miri)]
    const SAMPLES: usize = 100;

    let rng = StdRng::from_seed([0; 32]);
    let sx = Uniform::new(0.0, 1.0).unwrap();
    let sy = Uniform::new(0.0, 1.0).unwrap();
    let sxy = Product2 { dx: sx, dy: sy, f: |x, y| (x, y), pd: PhantomData::default() };
    let spi = Concentrated { dx: sxy, z: (0, 0), fa: |i_o, (x, y)| {
      if x*x + y*y < 1.0 { i_o.0 += 1 } else { i_o.1 += 1 }
      if i_o.0 + i_o.1 > SAMPLES { Some(4f64*(i_o.0 as f64/(i_o.0 + i_o.1) as f64)) } else { None }
    }, pd: Default::default() };

    spi.sample_iter(rng).take(10).for_each(|api| {
      let err_bar = 3.5f64 / (SAMPLES as f64).sqrt();
      assert!(std::f64::consts::PI-err_bar <= api && std::f64::consts::PI+err_bar >= api)
    });
  }

  #[test]
  fn categorical_samples() {
    #[cfg(not(miri))]
    const SAMPLES: usize = 1000;
    #[cfg(miri)]
    const SAMPLES: usize = 141;

    let rng = StdRng::from_seed([0; 32]);
    let expected = [('b', 2usize), ('a', 10), ('c', 29), ('d', 100)];
    let cd = ratios(expected.into_iter());
    let hist = Histogram::from_iter(cd.sample_iter(rng).take(SAMPLES*(10+2+29+100)));
    let achieved: Vec<(char, usize)> = hist.iter().map(|(k, c)|
      (*k, ((c as f64)/(SAMPLES as f64)).round() as usize)).collect();
    assert_eq!(&expected[..], &achieved[..]);
  }
}
