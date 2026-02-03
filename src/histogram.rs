use indexmap::IndexMap;

/// A convenience to analyze the distribution of generated values
#[derive(Debug)]
pub struct Histogram<T: std::cmp::Eq + std::hash::Hash> {
  pub lookup: IndexMap<T, usize>
}
impl <T: std::cmp::Eq + std::hash::Hash> FromIterator<T> for Histogram<T> {
  fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
    let mut lookup = IndexMap::new();
    for t in iter {
      *lookup.entry(t).or_insert(0) += 1;
    }
    // Sort by count (ascending)
    lookup.sort_by(|_, v1, _, v2| v1.cmp(v2));
    Histogram { lookup }
  }
}

impl <T: std::cmp::Eq + std::hash::Hash> Histogram<T> {
  pub fn from(iter: impl IntoIterator<Item=T>) -> Self {
    Self::from_iter(iter)
  }

  /// Returns an iterator over the histogram entries, sorted by occurrence count in ascending order.
  pub fn iter(&self) -> impl Iterator<Item = (&T, usize)> {
    self.lookup.iter().map(|(k, v)| (k, *v))
  }
}

impl<'a, T: std::cmp::Eq + std::hash::Hash> IntoIterator for &'a Histogram<T> {
  type Item = (&'a T, usize);
  type IntoIter = std::iter::Map<indexmap::map::Iter<'a, T, usize>, fn((&'a T, &'a usize)) -> (&'a T, usize)>;

  fn into_iter(self) -> Self::IntoIter {
    self.lookup.iter().map(|(k, v)| (k, *v))
  }
}

impl<T: std::cmp::Eq + std::hash::Hash> IntoIterator for Histogram<T> {
  type Item = (T, usize);
  type IntoIter = indexmap::map::IntoIter<T, usize>;

  fn into_iter(self) -> Self::IntoIter {
    self.lookup.into_iter()
  }
}
