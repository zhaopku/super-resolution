import os

def construct_dir(prefix, args):
	path = ''
	path += str(args.model)
	path += '_' + str(args.category)
	path += '_lr_'
	path += str(args.lr)
	path += '_bt_'
	path += str(args.batch_size)
	path += '_cr_' + str(args.crop_size)
	path += '_up_' + str(args.upscale_factor)
	if args.model == "SRGAN":
		path += '_gamma_' + str(args.gamma)
		path += '_theta_' + str(args.theta)
		path += '_sigma_' + str(args.sigma)

	if args.ratio != 1.0:
		path += '_ratio_' + str(args.ratio)

	return os.path.join(prefix, path)