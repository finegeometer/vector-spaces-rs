//! 
//! ``` [no_run]
//! use linear::*;
//! use nalgebra::{U2, U3};
//! struct Texture;
//! impl Space for Texture {
//! 	type Dim = U2;
//! }
//! 
//! struct World;
//! impl Space for World {
//! 	type Dim = U3;
//! }
//! 
//! struct Screen;
//! impl Space for Screen {
//! 	type Dim = U3;
//! }
//! 
//! let tex_to_world: Projective<f64, Texture, World> = unimplemented!();
//! let project: Projective<f64, World, Screen> = unimplemented!();
//! 
//! let tex_to_screen: Projective<f64, Texture, Screen> = project * tex_to_world;
//! 
//! ```

mod impls;

use num_traits::{Zero, One};
use nalgebra::{Scalar, DefaultAllocator, allocator::{Allocator, Reallocator}, DimName};
use nalgebra as na;

pub struct Linear<N: Scalar, A: Space, B: Space>(pub na::MatrixMN<N, B::Dim, A::Dim>)
	where DefaultAllocator: Allocator<N, B::Dim, A::Dim>;

/// Mark a type as being a type-level name for a vector space.
pub trait Space where
	Self::Dim: DimName,
{
	/// The dimension of the vector space.
	type Dim;
}

impl Space for () {
	type Dim = na::U0;
}

impl<A: Space, B: Space> Space for (A, B) where A::Dim: na::DimNameAdd<B::Dim>,
{
	type Dim = na::DimNameSum<A::Dim, B::Dim>;
}



impl<N: Scalar, X: Space, A: Space, B: Space> Linear<N, X, (A, B)> where
	A::Dim: na::DimNameAdd<B::Dim>,
	DefaultAllocator: Allocator<N, <(A, B) as Space>::Dim, X::Dim>,
	DefaultAllocator: Reallocator<N, <(A, B) as Space>::Dim, X::Dim, A::Dim, X::Dim>,
	DefaultAllocator: Reallocator<N, <(A, B) as Space>::Dim, X::Dim, B::Dim, X::Dim>,
	<(A, B) as Space>::Dim: na::DimSub<A::Dim, Output = B::Dim>,
	<(A, B) as Space>::Dim: na::DimSub<B::Dim, Output = A::Dim>,
{
	pub fn first_output(self) -> Linear<N, X, A> {
		Linear(self.0.remove_fixed_rows::<B::Dim>(0))
	}
	pub fn second_output(self) -> Linear<N, X, B> {
		Linear(self.0.remove_fixed_rows::<A::Dim>(B::Dim::dim()))
	}
	pub fn combine_outputs(a: Linear<N, X, A>, b: Linear<N, X, B>) -> Self {
		Linear(na::MatrixMN::<N, <(A, B) as Space>::Dim, X::Dim>::from_fn(|r, c| {
			if r < A::Dim::dim() {
				a.0[(r,c)]
			} else {
				b.0[(r - A::Dim::dim(), c)]
			}
		}))
	}
}

impl<N: Scalar, X: Space, A: Space, B: Space> Linear<N, (A, B), X> where
	A::Dim: na::DimNameAdd<B::Dim>,
	DefaultAllocator: Allocator<N, X::Dim, <(A, B) as Space>::Dim>,
	DefaultAllocator: Reallocator<N, X::Dim, <(A, B) as Space>::Dim, X::Dim, A::Dim>,
	DefaultAllocator: Reallocator<N, X::Dim, <(A, B) as Space>::Dim, X::Dim, B::Dim>,
	<(A, B) as Space>::Dim: na::DimSub<A::Dim, Output = B::Dim>,
	<(A, B) as Space>::Dim: na::DimSub<B::Dim, Output = A::Dim>,
{
	pub fn first_input(self) -> Linear<N, A, X> {
		Linear(self.0.remove_fixed_columns::<B::Dim>(0))
	}
	pub fn second_input(self) -> Linear<N, B, X> {
		Linear(self.0.remove_fixed_columns::<A::Dim>(B::Dim::dim()))
	}
	pub fn combine_inputs(a: Linear<N, A, X>, b: Linear<N, B, X>) -> Self {
		Linear(na::MatrixMN::<N, X::Dim, <(A, B) as Space>::Dim>::from_fn(|r, c| {
			if c < A::Dim::dim() {
				a.0[(r,c)]
			} else {
				b.0[(r, c - A::Dim::dim())]
			}
		}))
	}
}


/// Homogeneous coordinates
pub struct Homogeneous<A>(A);

impl<A: Space> Space for Homogeneous<A> where A::Dim: na::DimNameAdd<na::U1>,
{
	type Dim = na::DimNameSum<A::Dim, na::U1>;
}

pub type Projective<N, A, B> = Linear<N, Homogeneous<A>, Homogeneous<B>>;

impl<N: Scalar + Zero + One, A: Space, B: Space> Linear<N, A, B> where
	A::Dim: na::DimNameAdd<na::U1>,
	B::Dim: na::DimNameAdd<na::U1>,
	DefaultAllocator: Reallocator<N, B::Dim, A::Dim, <Homogeneous<B> as Space>::Dim, <Homogeneous<A> as Space>::Dim>,
{
	pub fn make_projective(self) -> Projective<N, A, B> {
		let mut mat = self.0.fixed_resize(N::zero());
		mat[(B::Dim::dim(), A::Dim::dim())] = N::one();
		Linear(mat)
	}
}
