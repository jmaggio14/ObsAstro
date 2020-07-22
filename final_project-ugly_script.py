import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal
import csv
import cv2
import os
from astropy.io.fits import getheader


# =============== global variables ===============

# filenames for this session
SOLVEPHT_OUT = 'solvepht_11.out'
SOLVEPHT_IMG = 'solvepht_11.img'
SOLVEPHT_SIG = 'solvepht_11.sig'
EXAMPLE_IMAGE = r"C:\Users\jmagg\Desktop\School-Online-Spring\test_image.png"
TARGET_COORDINATES = (957.6, 630)
REFERENCE_COORDINATES = (957, 413) # A
# REFERENCE_COORDINATES = (1014,792) # B
# REFERENCE_COORDINATES = (955, 502) # C
# REFERENCE_COORDINATES = (1014,792)#B
# PLOT_MAG_CUTOFF = 4.0
NSIGMA = 1.5
MAG_SHIFT = 10.430
JD_OFFSET = -2458691




def get_JD_date(fname):
	basename = os.path.splitext( os.path.basename(fname) )[0] + '.fits'
	fits_path = os.path.join( r'C:\Users\jmagg\Desktop\School-Online-Spring\ObsAstro\final project data\aligned_clean', basename )

	header = getheader(fits_path)
	jd = header['JD']
	return jd



# defining the indices and types for each column in solvepht output files
OUT_COL_INDICES = [ ('star_index', int),
					('X', float),
					('Y', float),
					('img_index', float),
					('time', float),
					('true_mag', float),
					('corrected_mag', float),
					('corrected_mag_err', float),
					('is_variable', int),
					('var_score', float),
					 ]


IMG_COL_INDICES = [ ('img_index', int),
					('fname', str),
					('exp_time', float),
					('time', float),
					('mag_zero', float),
					('z1', float),
					('z2', float),
					]

SIG_COL_INDICES = [ ('star_index', int),
					('true_mag', float),
					('true_mag_err', float),
					('is_variable', int),
					('var_score', float),
					('X', float),
					('Y', float),
					]

# ============== read in the files into their own dictionaries ==============

# OUT FILE
OUT_DATA = { field : [] for field,_ in OUT_COL_INDICES}
with open(SOLVEPHT_OUT, 'r') as f:
	# make csv reader object for this file
	out_reader = csv.reader(f, delimiter=' ')
	for row in out_reader:
		# scrub empty strings from row (fallout from bad delimination)
		row[:] = [c for c in row if c != '']
		for i,col in enumerate(row):
			# save the colum to the correct dict key and cast to the appropriate datatype
			field, dtype = OUT_COL_INDICES[i]
			OUT_DATA[field].append( dtype(col) )

print('read in',SOLVEPHT_OUT)

# IMG FILE
IMG_DATA = { field : [] for field,_ in IMG_COL_INDICES}
with open(SOLVEPHT_IMG, 'r') as f:
	# make csv reader object for this file
	img_reader = csv.reader(f, delimiter=' ')
	for row in img_reader:
		# scrub empty strings from row (fallout from bad delimination)
		row[:] = [c for c in row if c != '']
		for i,col in enumerate(row):
			# save the colum to the correct dict key and cast to the appropriate datatype
			field, dtype = IMG_COL_INDICES[i]
			IMG_DATA[field].append( dtype(col) )

print('read in',SOLVEPHT_IMG)


# SIG FILE
SIG_DATA = { field : [] for field,_ in SIG_COL_INDICES}
with open(SOLVEPHT_SIG, 'r') as f:
	# make csv reader object for this file
	sig_reader = csv.reader(f, delimiter=' ')
	for row in sig_reader:
		# scrub empty strings from row (fallout from bad delimination)
		row[:] = [c for c in row if c != '']
		for i,col in enumerate(row):
			# save the colum to the correct dict key and cast to the appropriate datatype
			field, dtype = SIG_COL_INDICES[i]
			SIG_DATA[field].append( dtype(col) )

print('read in',SOLVEPHT_SIG)




IMAGES_BY_INDEX = dict( zip(IMG_DATA['img_index'], IMG_DATA['fname']) )
INDEXES_BY_IMAGE = dict( zip(IMG_DATA['fname'], IMG_DATA['img_index']) )

IMAGE_INDEX_BY_TIME = [INDEXES_BY_IMAGE[fname] for fname in sorted(IMG_DATA['fname'])]
IMAGES_TIMES = { idx : get_JD_date(fname) for idx,fname in zip(IMAGE_INDEX_BY_TIME,sorted(IMG_DATA['fname'])) }

# IMAGE_SHIFTS = dict( zip(IMG_DATA['img_index'],(IMG_DATA['X'] )



# find target star index
TARGET_CANIDATES = []
REFERENCE_CANDIDATES = []
for idx,x,y,mag,err in zip(SIG_DATA['star_index'], SIG_DATA['X'], SIG_DATA['Y'], SIG_DATA['true_mag'], SIG_DATA['var_score']):
	if abs(x-TARGET_COORDINATES[0]) < 3 and abs(y-TARGET_COORDINATES[1]) < 3:
		TARGET_CANIDATES.append( (idx,x,y,mag,err) )

	if abs(x-REFERENCE_COORDINATES[0]) < 3 and abs(y-REFERENCE_COORDINATES[1]) < 3:
		REFERENCE_CANDIDATES.append( (idx,x,y,mag,err) )

# find reference star index


for tar in TARGET_CANIDATES:
	print("target index:", TARGET_CANIDATES[0][0])


# get all stellar magnitudes by stellar index'
STELLAR_MAGS = {star_idx : ([],[]) for star_idx in OUT_DATA['star_index']}
for star_idx,img_idx,mag in zip(OUT_DATA['star_index'], OUT_DATA['img_index'], OUT_DATA['corrected_mag']):
	t = IMAGES_TIMES[img_idx]
	STELLAR_MAGS[star_idx][0].append(t)
	STELLAR_MAGS[star_idx][1].append(mag)

# grab some metadata about the stars
STELLAR_DATA = {}
for star_idx,is_var,x,y in zip(OUT_DATA['star_index'],OUT_DATA['is_variable'],OUT_DATA['X'],OUT_DATA['Y']):
	STELLAR_DATA[star_idx] = {'is_variable':is_var, 'X':x, 'Y':y}


is_variable = np.asarray(SIG_DATA['is_variable']).astype(bool)
not_variable = np.invert(is_variable)


# ============= construct the ensemble plot =============
fig = plt.figure()

# sig mag plot
# -------------------------------------------------------------
ax = plt.subplot(111)
ax.set_xlabel('mean ensemble magnitude', fontsize=20)
ax.set_ylabel('std deviation from mean magnitude', fontsize=20)
# ax.set_title('ensemble uncertainty of stars')
ax.set_ylim(top=1)
ax.tick_params(labelsize=16)

is_var = []
is_target = []
mean_mags = []
scatter = []
for star_idx, (t,mags) in STELLAR_MAGS.items():
	mean_mag = np.mean(mags)
	if mean_mag < 0.5:
		print('saturated star: ', star_idx)

	mean_mags.append( mean_mag )
	scatter.append( np.std(mags) )
	is_var.append(STELLAR_DATA[star_idx]['is_variable'])
	is_target.append( star_idx in [tc[0] for tc in TARGET_CANIDATES] )

is_var = np.asarray(is_var).astype(bool)
not_var = np.invert(is_var)
is_target = np.asarray(is_target).astype(bool)
mean_mags = np.asarray(mean_mags)
scatter = np.asarray(scatter)

print('noise floor is:', np.mean(scatter[not_var]))


# # sort 'variable' stars from non-variable
# true_mags = np.asarray(SIG_DATA['true_mag'])
# var_score = np.asarray(SIG_DATA['var_score'])
#
# target_mags = [tar[3] for tar in TARGET_CANIDATES]
# target_errs = [tar[4] for tar in TARGET_CANIDATES]

# plot all non-variable stars
ax.scatter(mean_mags[not_var], scatter[not_var], marker='.', color='b')

# plot black pluses for variable stars
ax.scatter(mean_mags[is_var], scatter[is_var], marker='+', color='k')

# circle the target stars in red circles
ax.scatter(mean_mags[is_target], scatter[is_target], s=75, marker='o', color='r', facecolors='none')

# image correction? mag plot
# -------------------------------------------------------------
fig2 = plt.figure()


# sort images and magnitudes by filename (effectively by time)
mag_dict = dict(zip(IMG_DATA['fname'],IMG_DATA['mag_zero']))
# z2_dict = dict(zip(IMG_DATA['fname'],IMG_DATA['z2']))

mags = []
times = []
# z2s = []
for fname in sorted(IMG_DATA['fname']):
	mag = mag_dict[fname]
	# if mag > -14.6:
	# 	print('some bad images are:', INDEXES_BY_IMAGE[fname] )


	times.append( get_JD_date(fname) + JD_OFFSET )
	mags.append( mag_dict[fname] )
	# z2s.append( z2_dict[fname] )

ax1 = plt.subplot(111)
ax1.set_xlabel('start of exposure (JD %s)' % JD_OFFSET, fontsize=20)
ax1.set_ylabel('magnitude reference point', fontsize=20)
# ax1.set_title('Magnitude Correction Required for each Image', fontsize=24)
ax1.tick_params(labelsize=30)
ax1.scatter(times, mags)

# ax2 = plt.subplot(122)
# ax2.set_xlabel('image index')
# ax2.set_ylabel('zero point uncertainty')
# ax2.set_title('rms zero point / sqrt(N)')
# ax2.scatter(range(len(z2s)), z2s)


# plot variable stars on image
# -------------------------------------------------------------
fig3 = plt.figure()
ax = plt.subplot(111)
ax.set_xticks([])
ax.set_yticks([])

test_image = cv2.imread(EXAMPLE_IMAGE)
# example_image_idx = os.path.splitext(os.path.basename(EXAMPLE_IMAGE))[0]

LUT = np.linspace(255,0,256).astype(np.uint8)

plt.imshow(np.flip(LUT[test_image],axis=0))

# plot O's over variable stars
variable_x = np.asarray(SIG_DATA['X'])[is_variable]
variable_y = np.asarray(SIG_DATA['Y'])[is_variable]

# make every star that is greater than


non_variable_x = np.asarray(SIG_DATA['X'])[not_variable]
non_variable_y = np.asarray(SIG_DATA['Y'])[not_variable]

# ax.scatter(variable_x, variable_y, color='r', marker='o', facecolors='none')
ax.scatter(non_variable_x, non_variable_y, s=75, linewidths=2, color='b', marker='o', facecolors='none')

# draw green circles around possible targets
for tar,x,y,_,_ in TARGET_CANIDATES:
	ax.scatter(x, y,
				 s=100,
				 marker='s',
				 color='r',
				 linewidths=3,
				 facecolors='none')



# generate a light curve using ensemble analysis
# ---------------------------------------------------

fig4 = plt.figure()
ax = plt.subplot(111)
ax.set_title("Light Curve for TCP J21040470+4631129", fontsize=24)
ax.set_xlabel("JD-2458691", fontsize=24)
ax.set_ylabel("differential magnitude (V)", fontsize=24)
ax.tick_params(labelsize=16)






# iterate through out_data and fetch all data pertaining to our target

target_idx = TARGET_CANIDATES[0][0]
reference_idx = REFERENCE_CANDIDATES[0][0]

target_data = {}
ref_data = {}
err_data = {}
for star_idx,img_idx,mag,err in zip(OUT_DATA['star_index'], OUT_DATA['img_index'], OUT_DATA['corrected_mag'], OUT_DATA['corrected_mag_err']):
	if star_idx == target_idx:
		target_data[img_idx] = mag
		err_data[img_idx] = err

	if star_idx == reference_idx:
		ref_data[img_idx] = mag


# fetch the target mags and times for all images
times = []
tar_mags = []
ref_mags = []
errs = []

for img_idx in IMAGE_INDEX_BY_TIME:
	time = get_JD_date( IMAGES_BY_INDEX[img_idx] ) + JD_OFFSET

	tar_mag = target_data[img_idx] + MAG_SHIFT
	ref_mag = ref_data[img_idx] + MAG_SHIFT
	err_mag = err_data[img_idx]

	if tar_mag > 13.3:
		print('ananomlous mag for img', img_idx)

	times.append(time)
	tar_mags.append(tar_mag)
	ref_mags.append(ref_mag)
	errs.append(err_mag)





series1 = ax.scatter(times, tar_mags, marker='.', color='b')
# ax = plt.subplot(212)
series2 = ax.scatter(times, ref_mags, marker='x', color='r')

################################################################################
# fit cosine to the curve
def cosine(x, amplitude, period, phase, baseline):
	return (amplitude * np.cos((2*np.pi*x/period) + phase)) + baseline

def sawtooth(x, amplitude, period, phase, baseline, symmetry):
	return amplitude * signal.sawtooth((2*np.pi*x/period) + phase, symmetry) + baseline

def sym_sawtooth(x, amplitude, period, phase, baseline):
	return amplitude * signal.sawtooth((2*np.pi*x/period) + phase, 0.5) + baseline

def combo(x, a1, a2, period, ph1, ph2, baseline, symmetry):
	combo = cosine(x, a1, period, ph1, 0) + sawtooth(x, a2, period, ph2, -a2, symmetry)
	return combo + baseline

print('fitting cosine...')
(height, period, phase_shift, baseline), cov_mat = optimize.curve_fit(cosine,
															xdata=np.asarray(times),
															ydata=np.asarray(tar_mags),
															p0=(0.15,80/1440,0.64,13.12))

(height2, period2, phase_shift2, baseline2, sym2), cov_mat2 = optimize.curve_fit(sawtooth,
															xdata=np.asarray(times),
															ydata=np.asarray(tar_mags),
															p0=(0.15,80/1440,0.64,13.12,0.5))

param_error = np.sqrt( np.diag(cov_mat2) )
print('asymmetric sawtooth amplitude:', height2, '+/-', param_error[0], 'mag')
print('asymmetric sawtooth period:', period2*1440, '+/-', param_error[1]*1440, 'minutes')
print('asymmetric sawtooth phase_shift:', phase_shift2*1440, '+/-', param_error[2]*1440, 'minutes')
print('asymmetric sawtooth baseline:', baseline2, '+/-', param_error[3], 'mag')
print('asymmetric sawtooth symmetry:', sym2, '+/-', param_error[4], )




(height3, period3, phase_shift3, baseline3), cov_mat3 = optimize.curve_fit(sym_sawtooth,
															xdata=np.asarray(times),
															ydata=np.asarray(tar_mags),
															p0=(0.15,80/1440,0.64,13.12))

(height4_a, height4_b, period4, phase_shift4_a, phase_shift4_b, baseline4, sym4), cov_mat4 = optimize.curve_fit(combo,
															xdata=np.asarray(times),
															ydata=np.asarray(tar_mags),
															p0=(0.1,0.1,80/1440,0.64,0.64,12.5,0.5))

# plot cosine alongside light-curve
best_fit_cosine = cosine(      np.asarray(times), height, period, phase_shift, baseline)
best_fit_tri =    sawtooth(    np.asarray(times), height2, period2, phase_shift2, baseline2, sym2)
best_fit_symtri = sym_sawtooth(np.asarray(times), height3, period3, phase_shift3, baseline3)
best_fit_combo =  combo(       np.asarray(times), height4_a, height4_b, period4, phase_shift4_a, phase_shift4_b, baseline4, sym4)


# calculate standard error
ste_cosine = np.sqrt( np.sum( (np.asarray(tar_mags) - best_fit_cosine)**2 ) / best_fit_cosine.size )
print("cosine standard error is ", round(ste_cosine,6))

ste_tri = np.sqrt( np.sum( (np.asarray(tar_mags) - best_fit_tri)**2 ) / best_fit_tri.size )
print("sawtooth standard error is ", round(ste_tri,6))

ste_symtri = np.sqrt( np.sum( (np.asarray(tar_mags) - best_fit_symtri)**2 ) / best_fit_symtri.size )
print("symmetrical sawtooth standard error is ", round(ste_symtri,6))

ste_combo = np.sqrt( np.sum( (np.asarray(tar_mags) - best_fit_combo)**2 ) / best_fit_combo.size )
print("combo standard error is ", round(ste_combo,6))



fig5 = plt.figure()
plt.title("Least Squares fit of light curve")
plt.xlabel('JD')

ax = plt.subplot(411)
ax.plot(times, best_fit_cosine, color='r',)
ax.scatter(times, tar_mags, color='b', marker='.')
ax.tick_params(labelsize=14)
ax.set_xticks([])
ax.set_title("best fit cosine", fontsize=20)

ax = plt.subplot(412)
ax.plot(times, best_fit_tri, color='r',)
ax.scatter(times, tar_mags, color='b', marker='.')
ax.tick_params(labelsize=14)
ax.set_xticks([])
ax.set_title("best fit asymmetric sawtooth", fontsize=20)
# ax.scatter(times, ref_mags, color='r',  marker='.')

ax = plt.subplot(413)
ax.plot(times, best_fit_symtri, color='r',)
ax.scatter(times, tar_mags, color='b', marker='.')
ax.tick_params(labelsize=14)
ax.set_xticks([])
ax.set_title("best fit symmetric sawtooth", fontsize=20)
# ax.scatter(times, ref_mags, color='r',  marker='.')

ax = plt.subplot(414)
ax.plot(times, best_fit_combo, color='r',)
ax.scatter(times, tar_mags, color='b', marker='.')
ax.tick_params(labelsize=14)
ax.set_xlabel("JD-2458691", fontsize=17)
ax.set_title("best sawtooth + cosine", fontsize=20)
# ax.scatter(times, ref_mags, color='r',  marker='.')

print('superhump height is:', abs(height2)*2, '+/-', param_error[0]*2, 'magnitudes')
print('superhump period is: ', period2 * 1440, '+/-', param_error[1]*1440, 'minutes')

print('superhump max:', np.max(tar_mags))
print('superhump min:', np.min(tar_mags))




fig6 = plt.figure()
ax = plt.subplot(111)
ax.set_title("Light Curve for TCP J21040470+4631129", fontsize=24)
ax.set_xlabel("JD-2458691", fontsize=24)
ax.set_ylabel("differential magnitude (V)", fontsize=24)
ax.tick_params(labelsize=16)

series1 = ax.scatter(times, tar_mags, marker='.', color='b')
# ax = plt.subplot(212)
series2 = ax.scatter(times, ref_mags, marker='x', color='r')
# series3 = ax.plot(times, best_fit_tri, color='k')




# plt.legend(handles=[series1,series2], labels=['target','refererence'], fontsize='xx-large')

# calculate error
mag_error = np.mean(errs)
print("magnitude standard deviation is: ", mag_error)


# save the corrected magnitude data as a csv
out = np.hstack( (np.asarray(times).reshape((len(times),1)),np.asarray(tar_mags).reshape((len(tar_mags),1))) )

np.savetxt('out.csv', out, delimiter=',')


plt.ion()
plt.show()
plt.pause(0.01)

import pdb; pdb.set_trace()
