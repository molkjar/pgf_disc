# %%
import numpy as np
from scipy.linalg import toeplitz

# %%  
def compute_pgf_vecs(s, N, L_cdf, nu_cdf, rho, phi, stat="prev", offspring="negbin"):
  if offspring == "negbin":
    def p(z, phi):
      return - phi * (np.log(phi + 1 - z) - np.log(phi))
  if offspring == "pois":
    def p(z, phi):
      return z - 1

  def q_1(z, s):
    return s[np.newaxis, :] * np.exp(z)

  if stat == "prev":
    def q_2(z, s):
      return np.exp(z)
  if stat == "ci":
    def q_2(z, s):
      return s[np.newaxis, :] * np.exp(z)

  A_H = np.tile(np.arange(0, N + 1), (N + 1, 1)).T
  B_H = toeplitz(np.arange(0, N + 1))
  ind_h = np.tril(A_H - B_H)
  H = 1 - L_cdf[1:][ind_h]

  A_L = A_H[0:N, 0:N]
  B_L = np.tile(np.arange(N, 0, -1), (N, 1))
  ind_rho = np.triu(A_L + B_L)
  ind_nu = np.triu(B_L)
  L = rho[ind_rho] * np.diff(nu_cdf)[ind_nu]

  J = np.diff(L_cdf)[ind_nu]

  F = np.zeros((N + 1, N + 1, len(s)), dtype=np.complex_)
  F[:, 0, :] = H[:, 0, np.newaxis] * q_1(0, s)

  for i in range(1, N + 1):
    B = p(F[i:(N + 1), 0:i, :], phi) * L[0:(N - i + 1), (N - i):N, np.newaxis]
    B = np.cumsum(B[:, ::-1, :], axis = 1)[:, ::-1, :]

    int_1 = H[i:(N + 1), i, np.newaxis] * q_1(B[:, 0], s)

    B_2 = np.concatenate((B[:, 1:i, :], np.zeros((B.shape[0], 1, B.shape[2]))), axis = 1)
    int_2 = np.sum(q_2(B_2, s) * J[0:(N - i + 1), (N - i):N, np.newaxis], axis = 1)

    F[i:(N + 1), i, :] = int_1 + int_2

  return np.diagonal(F)



# %%  
def compute_pgf(s, N, L_cdf, nu_cdf, rho, phi, stat="prev", offspring="negbin"):
  if offspring == "negbin":
    def p(z, phi):
      return - phi * (np.log(phi + 1 - z) - np.log(phi))
  if offspring == "pois":
    def p(z, phi):
      return z - 1

  def q_1(z, s):
    return s * np.exp(z)

  if stat == "prev":
    def q_2(z, s):
      return np.exp(z)
  if stat == "ci":
    def q_2(z, s):
      return s * np.exp(z)
  
  A_H = np.tile(np.arange(0, N + 1), (N + 1, 1)).T
  B_H = toeplitz(np.arange(0, N + 1))
  ind_h = np.tril(A_H - B_H)
  H = 1 - L_cdf[1:][ind_h]

  A_L = A_H[0:N, 0:N]
  B_L = np.tile(np.arange(N, 0, -1), (N, 1))
  ind_rho = np.triu(A_L + B_L)
  ind_nu = np.triu(B_L)
  L = rho2[ind_rho] * np.diff(nu_cdf)[ind_nu]

  J = np.diff(L_cdf)[ind_nu]

  F = np.zeros((N + 1, N + 1), dtype=np.complex_)
  F[:, 0] = H[:, 0] * q_1(0, s)

  for i in range(1, N + 1):
    B = p(F[i:(N + 1), 0:i], phi) * L[0:(N - i + 1), (N - i):N]
    B = np.cumsum(B[:, ::-1], axis = 1)[:, ::-1]

    int_1 = H[i:(N + 1), i] * q_1(B[:, 0], s)
    
    B_2 = np.concatenate((B[:, 1:i], np.zeros((B.shape[0], 1))), axis = 1)
    int_2 = np.sum(q_2(B_2, s) * J[0:(N - i + 1), (N - i):N], axis = 1)

    F[i:(N + 1), i] = int_1 + int_2

  return np.diag(F)


# %%
def compute_pgf_prep(N, Delta, G, V, rho, phi, count="prev", inf_proc="negbin"):
  if inf_proc == "negbin":
    def psi(z, phi):
      return - phi * (np.log(phi + 1 - z) - np.log(phi))
  if inf_proc == "pois":
    def psi(z, phi):
      return z - 1

  def q_1(z, s):
    return s * np.exp(z)

  if count == "prev":
    def q_2(z, s):
      return np.exp(z)
  if count == "ci":
    def q_2(z, s):
      return s * np.exp(z)

  Gbar_t = np.tril(np.tile(np.arange(0, N+1) * Delta, (N+1, 1)).T)
  Gbar_tau = np.tril(toeplitz(np.arange(0, N+1) * Delta))
  Gbar = 1 - G(Gbar_t - Gbar_tau, Gbar_tau)

  D_t = np.triu(np.tile(np.arange(N, -1, -1) * Delta, (N, 1)))
  D_tau = np.triu(np.tile(np.arange(0, N) * Delta, (N+1, 1)).T)
  
  DV = rho(D_t[:, :-1] + D_tau[:, :-1]) * (-1) * np.diff(V(D_t))
  DG = -np.diff(G(D_t, D_tau))

  return [N, psi, q_1, q_2, Gbar, DV, DG]  


def compute_pgf_fun(s, prep_list):
  N, psi, q_1, q_2, Gbar, DV, DG = prep_list
  
  F = np.zeros((N + 1, N + 1), dtype=np.complex_)
  F[:, 0] = s

  for i in range(1, N + 1):
    B = psi(F[i:(N + 1), 0:i], phi) * DV[0:(N - i + 1), (N - i):N]
    B_1 = np.cumsum(B[:, ::-1], axis = 1)[:, ::-1]

    int_1 = Gbar[i:(N + 1), i] * q_1(B_1[:, 0], s)
    
    B_2 = np.concatenate((B_1[:, 1:i], np.zeros((B_1.shape[0], 1))), axis = 1)
    int_2 = np.sum(q_2(B_2, s) * DG[0:(N - i + 1), (N - i):N], axis = 1)

    F[i:(N + 1), i] = int_1 + int_2

  return np.diag(F)










# %%
def compute_pmf(M, N):
    num = np.zeros((M, N + 1))+0j
    for m in range(0, M):
        num[m, :] = np.transpose(compute_pgf(np.exp(2.0*np.pi*1j*m/M), N, L_cdf, nu_cdf, rho, phi))
    
    fft = np.real(np.fft.fft(num, axis = 0))  
    pmf = fft * (fft >= 0)
    pmf_norm = pmf / np.sum(pmf, axis = 0)
    
    return pmf_norm

def compute_pmf2(M, N):
    num = np.zeros((M, N + 1))+0j
    p_list = compute_pgf_prep(N, 0.1, G, V, rho, phi, inf_proc="pois")
    for m in range(0, M):
        num[m, :] = np.transpose(compute_pgf_fun(np.exp(2.0*np.pi*1j*m/M), p_list))
    
    fft = np.real(np.fft.fft(num, axis = 0))  
    pmf = fft * (fft >= 0)
    pmf_norm = pmf / np.sum(pmf, axis = 0)
    
    return pmf_norm

def compute_pmf_vecs(M, N):
  num = compute_pgf_vecs(np.exp(2.0 * np.pi * 1j * np.arange(0, M) / M), N, L_cdf, nu_cdf, rho, phi)
  fft = np.real(np.fft.fft(num, axis = 0))  
  pmf = fft * (fft >= 0)
  pmf_norm = pmf / np.sum(pmf, axis = 0)

  return pmf_norm

def mean(pmf):
    M = np.shape(pmf)[0]
    loc = np.linspace(0, M-1, M)
    mn = np.sum(pmf.T * loc, axis = 1)
    return mn

def quick_mean(rho):
    N = np.shape(rho)[0]
    F = np.zeros((N + 1, N + 1))
    F[:, 0] = (1 - L_cdf[1:][0])
    for c in range(0, N):
        for t in range(1, c+1):
            convolution = 0
            for u in range(1, t+1):
                convolution += rho[c-t+u]*(1 - L_cdf[1:][u])*np.diff(nu_cdf)[u]*F[c, t-u]
            F[c, t] = (1 - L_cdf[1:][t]) + convolution
    return np.diag(F)

