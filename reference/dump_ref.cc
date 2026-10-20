// Reference dump harness: exports SPRITE intermediate results for unit tests
// of the Python port. Links against the unmodified SPRITE v1.0 libraries.
#include <string>
#include <cstdio>
#include <cstdlib>
#define private public
#include "sprite.h"
#undef private
#include "sr_util.h"
#include "IM_IO.h"
#include "IM3D_IO.h"

static std::string OUT;

static void wI(const std::string &name, Ifloat &I)
{
  std::string p = OUT + name;
  io_write_ima_float((char *)p.c_str(), I);
}

static void wA(const std::string &name, fltarray &A)
{
  std::string p = OUT + name;
  io_3d_write_data((char *)p.c_str(), A);
}

// Bands stored with the same convention as the SPRITE code: A(i,j,b) = MR(b,j,i)
static void wMR(const std::string &name, MultiResol &MR)
{
  int nb = MR.nbr_band(), nc = MR.size_band_nc(0), nl = MR.size_band_nl(0);
  fltarray A(nc, nl, nb);
  for (int b = 0; b < nb; b++)
    for (int i = 0; i < nc; i++)
      for (int j = 0; j < nl; j++) A(i, j, b) = MR(b, j, i);
  wA(name, A);
}

static void wtxt(const std::string &name, const double *v, int n)
{
  std::string p = OUT + name;
  FILE *f = fopen(p.c_str(), "w");
  for (int k = 0; k < n; k++) fprintf(f, "%.17g\n", v[k]);
  fclose(f);
}

// Deterministic pseudo-random generator (inputs are dumped as rnd_*.fits)
static unsigned long long lcg_state = 12345ULL;
static double lcg()
{
  lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return ((lcg_state >> 11) * (1.0 / 9007199254740992.0)) - 0.5;
}

int main(int argc, char *argv[])
{
  if (argc < 4)
    {
      fprintf(stderr, "usage: %s cube.fits out_dir/ transform_id(2|24)\n", argv[0]);
      return 1;
    }
  OUT = argv[2];
  int t = atoi(argv[3]);
  type_transform sr_transf = (type_transform)(t - 1);

  fltarray Dat;
  fitsstruct Header;
  io_3d_read_data(argv[1], Dat, &Header);
  int Nx = Dat.nx(), Ny = Dat.ny(), Nz = Dat.nz();
  int up = 2;

  // ---- Orientation probes ----
  Ifloat pI(4, 6);
  pI(1, 2) = 1;
  wI("probe_I.fits", pI);
  fltarray pA(4, 6, 3);
  pA(1, 2, 0) = 1;
  pA(3, 0, 2) = 2;
  wA("probe_A.fits", pA);

  // ---- Deterministic random inputs ----
  Ifloat rhr(Nx * up, Ny * up);
  for (int i = 0; i < Nx * up; i++)
    for (int j = 0; j < Ny * up; j++) rhr(i, j) = lcg();
  wI("rnd_hr.fits", rhr);
  fltarray rlr(Nx, Ny, Nz);
  for (int k = 0; k < Nz; k++)
    for (int i = 0; i < Nx; i++)
      for (int j = 0; j < Ny; j++) rlr(i, j, k) = lcg();
  wA("rnd_lr.fits", rlr);

  // ---- Lanczos kernel and FFT convolution ----
  int lrad = 10;
  Ifloat ker(2 * lrad + 1, 2 * lrad + 1);
  double u[2] = {0.3, -0.7};
  {
    double **tk = (double **)malloc((2 * lrad + 1) * sizeof(double *));
    for (int k = 0; k < 2 * lrad + 1; k++) tk[k] = (double *)malloc((2 * lrad + 1) * sizeof(double));
    lanczos(u, tk, lrad);
    for (int i = 0; i < 2 * lrad + 1; i++)
      for (int j = 0; j < 2 * lrad + 1; j++) ker(i, j) = tk[i][j];
    for (int k = 0; k < 2 * lrad + 1; k++) free(tk[k]);
    free(tk);
  }
  wI("lanczos_03_m07.fits", ker);
  Ifloat conv(Nx * up, Ny * up);
  psf_convol(rhr, ker, conv, True, False);
  wI("conv_hr.fits", conv);
  Ifloat ker_rot(2 * lrad + 1, 2 * lrad + 1);
  rotate(&ker, &ker_rot, 2);
  wI("lanczos_rot2.fits", ker_rot);

  // ---- Decimation / transpose ----
  Ifloat filt(Nx * up, Ny * up), dec(Nx, Ny);
  decim(&rhr, &filt, &dec, up, &ker, True, True);
  wI("decim_hr.fits", dec);
  Ifloat lr0(Nx, Ny), tdec(Nx * up, Ny * up);
  for (int i = 0; i < Nx; i++)
    for (int j = 0; j < Ny; j++) lr0(i, j) = rlr(i, j, 0);
  transpose_decim(&lr0, &tdec, up);
  wI("transpose_decim.fits", tdec);

  // ---- Options exactly as in run_sprite main ----
  int siz = MIN(Nx, Ny);
  mr_opt sr_opt = mr_opt_init(siz * up, sr_transf, -1, -1, NORM_L1, DEF_UNDER_FILTER, F_MALLAT_7_9, I_CONT, False);
  mr_opt mod_opt = mr_opt_init(siz * up, TO_UNDECIMATED_MALLAT, -1, -1, NORM_L1, DEF_UNDER_FILTER, F_MALLAT_7_9, I_CONT, False);
  mr_opt den_opt = mr_opt_init(Nx, TO_UNDECIMATED_MALLAT, 2, 2, NORM_L2);

  // ---- Wavelet transforms ----
  {
    MultiResol MR;
    wl_trans(&rhr, sr_opt, &MR);
    wMR("wt_sr.fits", MR);
    Ifloat rec(Nx * up, Ny * up);
    MR.recons(rec, sr_opt.Bord);
    wI("wt_sr_rec.fits", rec);
    fltarray nmap(Nx * up, Ny * up, MR.nbr_band() - 1);
    wl_gnoise_est(&MR, &nmap);
    wA("wt_sr_gnoise.fits", nmap);
    MR.free();
  }
  {
    MultiResol MR;
    wl_trans(&rhr, mod_opt, &MR);
    wMR("wt_mod.fits", MR);
    Ifloat rec(Nx * up, Ny * up);
    MR.recons(rec, mod_opt.Bord);
    wI("wt_mod_rec.fits", rec);
    MR.free();
  }
  {
    MultiResol MR;
    wl_trans(&lr0, den_opt, &MR);
    wMR("wt_den.fits", MR);
    MR.free();
  }
  {
    fltarray nmap(Nx, Ny, Nz);
    noise_est(&Dat, &nmap);
    double s[64];
    for (int k = 0; k < Nz; k++) s[k] = nmap(0, 0, k);
    wtxt("noise_est.txt", s, Nz);
  }

  // ---- Analysis prox (wl_filter_analysis) on a random image ----
  {
    MultiResol MR;
    int nb_band = MR.computeNbBand(Nx * up, Ny * up, sr_opt.nb_sc, sr_opt.transf, sr_opt.nb_usc);
    fltarray std_map(Nx * up, Ny * up, nb_band - 1);
    for (int k = 0; k < nb_band - 1; k++)
      for (int i = 0; i < Nx * up; i++)
        for (int j = 0; j < Ny * up; j++) std_map(i, j, k) = 0.02 * (k + 1);
    MultiResol coeff_init;
    coeff_init.alloc(Ny * up, Nx * up, sr_opt.nb_sc, sr_opt.transf);
    Ifloat out(Nx * up, Ny * up);
    wl_filter_analysis(rhr, out, sr_opt, 4.0, True, std_map, coeff_init, 100);
    wI("prox_analysis.fits", out);
    wMR("prox_analysis_coeff.fits", coeff_init);
  }

  // ---- Sprite object, constructed exactly as in run_sprite (-F -N -s 4) ----
  string rep = OUT, of = "solution.fits", rf = "resi.fits";
  Sprite S(up, &Dat, mod_opt, sr_opt, 50, MR_SUPP_DENOISING, 4.0, rep, of, rf, 0, ACCURATE_ANALYSIS, GEN_FWBW,
           False, False, 25, 0.005, False, NULL, True, NULL, 0, MODEL_SINC_INTERP, True, 0, True);
  wI("model.fits", S.model);
  wA("noise_sig_map.fits", S.noise_sig_map);
  wA("data_nrm.fits", S.data_nrm);
  wA("lanczos_ker.fits", S.lanczos_ker);
  {
    double v[64];
    for (int k = 0; k < Nz; k++) v[k] = S.cent[0][k];
    wtxt("cent0.txt", v, Nz);
    for (int k = 0; k < Nz; k++) v[k] = S.cent[1][k];
    wtxt("cent1.txt", v, Nz);
    for (int k = 0; k < Nz; k++) v[k] = S.shift[0][k];
    wtxt("shift0.txt", v, Nz);
    for (int k = 0; k < Nz; k++) v[k] = S.shift[1][k];
    wtxt("shift1.txt", v, Nz);
    for (int k = 0; k < Nz; k++) v[k] = S.flux[k];
    wtxt("flux.txt", v, Nz);
    v[0] = S.sig_max;
    wtxt("sig_max.txt", v, 1);
  }

  // Stacked first guess before multiresolution-support denoising
  {
    Ifloat stack(Nx * up, Ny * up);
    lanczos_stacking(Dat, stack, up, S.shift_cent, 5);
    wI("stack.fits", stack);
    Ifloat filt2(Nx * up, Ny * up);
    mr_support_filt(stack, filt2, mod_opt, 5);
    wI("stack_mrsupp.fits", filt2);
  }

  // Forward and adjoint operators on random inputs
  {
    fltarray out(Nx, Ny, Nz);
    S.sprite_op_mat(rhr, out);
    wA("op_M.fits", out);
    Ifloat outT(Nx * up, Ny * up);
    S.sprite_transp_op_mat(rlr, outT);
    wI("op_MT.fits", outT);
  }

  // Spectral radius (rand() is unseeded, hence deterministic)
  {
    int nit;
    S.spectral_rad_est(&nit);
    double v[2] = {S.spectral_rad, (double)nit};
    wtxt("spectral_rad.txt", v, 2);
  }

  // PSF size used for centroid weighting and flux aperture (calls mod_gen: done last)
  {
    double v = S.lr_psf_size();
    wtxt("sig_lr.txt", &v, 1);
    wI("lr_psf_model.fits", S.model);
    Ifloat coadd(Nx * up, Ny * up);
    coadd_frames(Dat, coadd, (float)up);
    wI("lr_psf_coadd.fits", coadd);
    double g[5];
    gauss2D_fitting_first_guess(S.model, &g[0], &g[1], &g[2], &g[3], &g[4], 3);
    wtxt("lr_psf_gauss_first_guess.txt", g, 5);
    gauss2d_fit_2(S.model, &g[2], &g[3], &g[4], &g[0], &g[1]);
    wtxt("lr_psf_gauss_fit.txt", g, 5);
  }
  printf("dump done\n");
  return 0;
}
