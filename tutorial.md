Tutorial    {#tutorial}
========

The following is a short tutorial illustrating the basic usage using the
quantum Ising model in one dimension as an example.  The quantum Ising model in
one-dimension has a Hamiltonian of the form

\f[
H = -J \sum_{\langle i ,j\rangle} \sigma_i^z \sigma_j^z - h \sum_{i} \sigma_i^x
\f]

I have tried to make this tutorial self-contained. For a very accessible
introduction to the things we want to achieve, I warmly recommend the notes by
Norbert Schuch for his lecture 
["Analytical and Numerical Methods for Quantum Many-Body Systems from a Quantum Information Perspective"](https://www.quantuminfo.physik.rwth-aachen.de/go/id/ghwb/lidx/1).
Lectures 3 to 6 provide a good basis for the things I will be talking about.
Then there is of course also the review article by Ulrich Schollwoeck, 
["The density-matrix renormalization group in the age of matrix product states"](https://arxiv.org/abs/1008.3477), 
which is nice, although a bit dense
at times.



MPO representation of the Hamiltonian 
-------------------------------------

The first thing we have to do is to create a MPO (matrix product operator)
representation of the Hamiltonian. To see why that representation is useful,
consider the expectation value \f$\langle \psi | H | \psi \rangle\f$ of the
Hamiltonian in a state \f$|\psi\rangle\f$. Let us imagine that our state
\f$|\psi\rangle\f$ is a product state such that it can be written in the form

\f[
|\psi\rangle = \bigotimes_{s=1}^N |\psi_s\rangle, \quad |\psi_s\rangle = \sum_{\sigma=1}^d (A^{[s]})^{\sigma} |\sigma\rangle
\f]

The vector \f$A^{[s]}\f$ is the vector of coefficients of the state
\f$|\psi_s\rangle\f$ at site s expressed in the basis \f$|\sigma\rangle\f$. The physical dimension d
of this on-site Hilbert space is d=2 for the quantum Ising case.  With the product
state, the calculation of the expectation value of an arbitrary operator
factorizes. This means that we can construct the expectation value of any
tensor operator \f$ V = \bigotimes_s V^{[s]}\f$ by multiplying together the
expectation values \f$ \langle \psi_s | V^{[s]} | \psi_s\rangle\f$. Taking a
system with N=5 sites as an example, we can represent the calculation of the
expectation value graphically in the form 

![Graphical representation of the calculation of the expectation value](../img/mpo_expectation_value_product_state_simple.svg) 

Here, the lines between \f$A^{[s]}\f$ and \f$V^{[s]}\f$ represent the
contraction of the indices during the calculation of the expectation value. 

The original Hamiltonian is a sum of several different operators \f$V_j\f$.
The general idea of the MPO approach is to consider operator-valued matrices 
\f$W^{[s]}\f$ instead of single operators \f$V^{[s]}\f$. The matrices 
\f$W^{[s]}\f$ are constructed in such a way that we obtain the Hamiltonian H
by multiplying all of the matrices.  In order to construct the matrixes, we
take a look at all the possible operator strings that arise in the Hamiltonian.
Let's start with the rightmost operator at the last site. The operator
appearing there can either be the identity 1, the operator  \f$-h \sigma^x\f$
(then all the operators to the left are the identity) or \f$\sigma^z\f$ (then
the operator directly to the left of the last site must be \f$-J \sigma^z\f$ and
all other operators are the identity). We combine all possibilities into a
vector \f$W^{[N]}_{b_{N-1},1} = (1, \sigma^{z},  -h \sigma^x)_{b_{N-1}}\f$. Going
to the next site N-1, there are three possible states of the operator string
looking from site N-1 to the right (with site N-1 included): 
 
- Type 1: only identities.
- Type 2: \f$\sigma^z\f$ followed by identities.
- Type 3: completed interaction term \f$-J \sigma^z_{j} \sigma^z_{j+1}\f$ or
completed on-site term \f$-h \sigma^x\f$.

We introduce the matrix

\f[
W_{b_{s-1} b_s}^{[s]} = \begin{pmatrix}
1 & 0 & 0 \\
\sigma^z & 0 & 0 \\
-h \sigma^x & -J \sigma^z & 1 
\end{pmatrix}_{b_{s-1} b_s}
\f]

characterizing the mapping from either of the 3 configurations at site s+1 to
either of the 3 configurations at site s. With the matrix 
\f$W^{[1]}_{1,b_1} = (-h \sigma^x, -J \sigma^z,  1)_{b_1}\f$,
we can write the Hamiltonian in the form

\f[
H = \sum_{b_1, \dots, b_{N-1}} W_{1,b_1}^{[1]} \otimes W^{[2]}_{b_1,b_2} \otimes \dots \otimes W^{[N]}_{b_{N-1},1}
\f]

Using the MPO representation, we can represent the calculation of the
expectation value graphically in the form 

![Graphical representation of the calculation of the expectation value](../img/mpo_expectation_value_product_state.svg) 

Here, the horizontal lines between the different matrices \f$W^{[j]}\f$
symbolize the contraction of the matrix indices, i.e., a matrix product.

You might argue that this construction can barely be helpful since we restrict
ourselves to product states. However, note that later we will represent the
states of our system as matrix product states which are straightforward
generalizations of product states such that the MPO representation remains
useful.

Let us take a look at how we represent such a MPO in the code. We define our
own class IsingMPO derived from TransInvMPO.
 
    class IsingMPO : public TransInvMPO<double, 2, 2> 
    {
    public:
      IsingMPO(std::size_t N, double J);

      Matrix<double, 2, 2> pauliX;
      Matrix<double, 2, 2> pauliZ;
      Matrix<double, 2, 2> opEye;
    };

    IsingMPO::IsingMPO(std::size_t N, double J) : TransInvMPO<double, 2, 2>(N, 3) 
    {
      pauliX << 0, 1,
            1, 0;
      pauliZ << 1, 0,
            0, -1;
      opEye << 1, 0,
           0, 1;
      double h = 1;
 
      this->wBulk.setElement(0, 0, opEye, true);
      this->wBulk.setElement(1, 0, pauliZ);
      this->wBulk.setElement(2, 0, -h*pauliX);
      this->wBulk.setElement(2, 1, -J*pauliZ);
      this->wBulk.setElement(2, 2, opEye, true);

      this->wLeft.setElement(0, 0, -pauliZ);
      this->wLeft.setElement(0, 1, -h*pauliX);
      this->wLeft.setElement(0, 2, opEye, true);

      this->wRight.setElement(0, 0, opEye, true);
      this->wRight.setElement(1, 0, pauliZ);
      this->wRight.setElement(2, 0, -h*pauliX);
    }

TransInvMPO is derived from the class MPO and represents translationally
invariant MPOs. The first template argument specifies the data type (double in
this case) and the next two give the dimension of the on-site operators (i.e.,
the dimensions of the Pauli matrices). For the translationally invariant case,
we only have to define the variables wBulk, wLeft, and wRight which are
instances of MPOTensor. MPOTensor instances represent the W matrices.
MPOTensor is a sparse data structure storing operators. 


Variational ground-state search
-------------------------------

Let us next look at how we perform the variational ground-state search.
With the MPO representation of our Hamiltonian set up, we have already done 
most of the necessary work. 

    // Internal bond dimension of the MPS
    std::size_t D = 32;
    // Size of the chain
    std::size_t N = 16;
    double J = 10.;
    // Create a MPS with N sites of physical dimension d=2 and internal bond dimension D
    MPS<double> mps(std::vector<std::size_t>(N, 2), D);
    // MPO representation of the Hamiltonian
    IsingMPO ham(N, J);
    // Square of the Hamiltonian as a MPO
    TransInvMPO<double, 2, 2> ham2 = ham.dot(ham);
    // Class performing the variational optimization of mps for the ground state search of ham 
    DMRGOptimizer<double, 2, 2> dmrg(mps, ham);
    // specifiy the numerical precision values to use in each sweep 
    double tols[] = {1e-4,1e-6,1e-8,1e-10};
    std::size_t ntols = 4;
    double en = 0;
    // perform as many sweeps as there are elements in tol
    for (std::size_t i = 0; i < ntols; ++i)
    {
      std::cout << "Performing run " << i << " at precision " << tols[i] 
        << std::endl;
      double tol = tols[i];
      en = dmrg.fullSweep(tol);
    }
    double en2 = mps.expValue(ham2);
    // get a variational estimate how close we are to a gound ground state
    // approximation
    double qual = std::sqrt(std::abs(en*en - en2))/std::abs(en);

    std::cout << "Qual is " << qual << std::endl;

    TransInvSingleSiteMPO<double, 2, 2> mpoMag(mps.N, 
      1./mps.N*ham.pauliZ);

    double mag = mps.expValue(mpoMag);
    
    std::cout << "mag: " << mag << std::endl;

    return 0;

Here, we create an MPS instance mps representing a state \f$|\psi\rangle\f$.
Its tensors are automatically initialized in a random way such that the
expectation value of the norm \f$|\langle \psi | \psi \rangle|\f$ is close to
one.  The double value tols[i] contains the numerical precision to use for
sweep i.  It improves performance if the numerical precision is slowly
increased with further sweeps. The expectation value \f$ \langle \psi | (H -
    \langle \psi | H | \psi \rangle)^2 | \psi\rangle\f$ is zero for an exact
eigenvector and therefore gives us an estimate how close we are to a true  
eigenstate.

### DMRG in a nutshell ### 

The DMRG algorithm optimizes the functional 

\f[
  f_\lambda(|\psi\rangle) = \langle \psi | H | \psi \rangle 
  - \lambda(\langle \psi | \psi\rangle - 1).
\f]

Here, the Lagrange multiplier \f$\lambda\f$ from the last term enforces the
state normalization. Minimizing the functional with respect to
\f$|\psi\rangle\f$ leads to the eigenvalue problem

\f[
  H |\psi\rangle = \lambda |\psi \rangle.
\f]

Since this eigenvalue problem involves the full Hilbert space of the many-body
problem, it is generally not tractable.  The DMRG algorithm circumvents the
problem by optizing iteratively each of the tensors \f$A^{[s]}\f$ at the sites \f$s = 1,\dots,N\f$
while keeping the other tensors fixed. Minimizing the functional with respect
to \f$(A^{*[s]})^{\sigma_s'}_{a_{s-1}'a_s'}\f$ at site s leads to

\f[
  \sum_{a_{s-1},a_s,\sigma_s} H_{a_{s-1}' a_s' a_{s-1} a_s}^{\sigma_s' \sigma_s} A_{a_{s-1} a_s}^{\sigma_s}
  = \lambda \sum_{a_{s-1},a_s,\sigma_s} S_{a_{s-1}' a_s' a_{s-1} a_s}^{\sigma_s' \sigma_s} A_{a_{s-1}
    a_s}^{\sigma_s}
\f]

where

\f[
  H_{a_{s-1}' a_s' a_{s-1} a_s}^{\sigma_s' \sigma_s} =
  \sum_{b_1,\dots,b_N} 
  \biggl(
    \sum_{a_1,\dots,a_{s-2} \\ a_1',\dots,a_{s-2}'} 
    \prod_{j=1}^{s-1} 
    \sum_{\sigma_j,\sigma_j'=1}^d (A^{[j]*})^{\sigma_j'}_{a_{j-1}'a_j'} 
    (W^{[s]})^{\sigma_j' \sigma_j}_{b_{s-1} b_s}
    (A^{[j]})^{\sigma_j}_{a_{j-1}a_j}\biggr) 
  \biggl(
    \sum_{a_{s+1},\dots,a_{N} \\ a_{s+1}',\dots,a_{N}'} 
    \prod_{j=s+1}^{N} 
    \sum_{\sigma_j,\sigma_j'=1}^d (A^{[j]*})^{\sigma_j'}_{a_{j-1}'a_j'} 
    (W^{[s]})^{\sigma_j' \sigma_j}_{b_{s-1} b_s}
    (A^{[j]})^{\sigma_j}_{a_{j-1}a_j}\biggr)  

\f]

and

\f[
  S_{a_{s-1}' a_s' a_{s-1} a_s}^{\sigma_s' \sigma_s} = 
  \biggl( 
    \sum_{a_1,\dots,a_{s-2} \\ a_1',\dots,a_{s-2}'} 
    \prod_{j=1}^{s-1} 
    \sum_{\sigma_j=1}^d (A^{[j]*})^{\sigma_j}_{a_{j-1}'a_j'}
    (A^{[j]})^{\sigma_j}_{a_{j-1}a_j}\biggr) 
  \biggl( \sum_{a_{s+1},\dots,a_{N} \\ a_{s+1}',\dots,a_{N}'} 
    \prod_{j=s+1}^{N} \sum_{\sigma_j=1}^d 
    (A^{[j]*})^{\sigma_j}_{a_{j-1}'a_j'} 
    (A^{[j]})^{\sigma_j}_{a_{j-1}a_j}\biggr) 
  \delta_{\sigma_s', \sigma_s}.
\f]

It is much easier to see what these formulas mean using the graphical
representations below:

![Calculation of the expectation value of the Hamiltonian](../img/ham_expectation_value.svg)

![Calculation of the state normalization](../img/state_normalization.svg)

Considering \f$(\sigma_s, a_{s-1}, a_s)\f$ as a multiindex, we can consider
\f$(A^{[s]})^{\sigma_s}_{a_{s-1},a_s}\f$ as a vector v. With this interpretation,
  optimizing the tensor amounts to solving the generalized eigenvalue problem 

\f[ 
  H v = \lambda S v.
\f]. 

The DMRG algorithm proceeds by sweeping through the sites s of the system,
solving the eigenvalue problem at each site and replacing the tensor at site s
by the eigenvector to the smallest eigenvalue.


### Transformation into a conventional eigenvalue problem ###

The solution of this generalized eigenvalue problem is numerically unstable if
the the matrix S is badly conditioned. Exploiting the invariance under a gauge
transformation 

\f[
  (A^{[s]})^{\sigma_s}_{a_{s-1} a_s} 
  \rightarrow \sum_{a_{s-1}', a_s} X_{a_{s-1} a_{s-1}'} (A^{[s]})^{\sigma_s}_{a_{s-1}' a_{s}'} 
  X^{-1}_{a_s' a_s},
\f]

of all tensors with a matrix X, it is possible to bring the MPS into a
left-canonical form and a right-canonical form around site s. 

### Implementation details ###


