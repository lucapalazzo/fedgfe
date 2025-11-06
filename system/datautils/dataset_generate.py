import sys
from datautils.generate_cifar10 import generate_cifar10
from datautils.generate_mnist import generate_mnist
from datautils.generate_cifar100 import generate_cifar100
from datautils.generate_chestxray import generate_chestxray
from datautils.generate_jsrt import generate_jsrt
from datautils.generate_darwin import generate_darwin
from datautils.generate_shenzen_montgomery import generate_shenzen_montgomery

def dataset_generate ( args ):
    if args.dataset == "mnist" or args.dataset == "fmnist":
        outdir = 'dataset/mnist/'
        if ( args.dataset_outdir != None ):
            outdir = 'dataset/' + args.dataset_outdir + '/'
        generate_mnist( args, outdir, args.num_clients, 10, args.dataset_niid, args.dataset_balance, args.dataset_partition, args.dataset_dir_alpha, class_per_client=args.num_classes_per_client)
    elif args.dataset == "CIFAR-10":
        outdir = 'dataset/CIFAR-10/'
        if ( args.dataset_outdir != None ):
            outdir = 'dataset/' + args.dataset_outdir + '/'
        generate_cifar10( args, outdir, args.num_clients, 10, args.dataset_niid, args.dataset_balance, args.dataset_partition, args.dataset_dir_alpha, class_per_client=args.num_classes_per_client)
    elif args.dataset == "CIFAR-100":
        outdir = 'dataset/CIFAR-100/'
        if ( args.dataset_outdir != None ):
            outdir = 'dataset/' + args.dataset_outdir + '/'
        generate_cifar100( args, outdir, args.num_clients, 100, args.dataset_niid, args.dataset_balance, args.dataset_partition, args.dataset_dir_alpha, class_per_client=args.num_classes_per_client)
    elif args.dataset == "ChestXRay":
        outdir = 'dataset/ChestXRay/'
        if ( args.dataset_outdir != None ):
            outdir = 'dataset/' + args.dataset_outdir + '/'
        generate_chestxray( args, outdir, args.num_clients, 100, args.dataset_niid, args.dataset_balance, args.dataset_partition, args.dataset_dir_alpha, class_per_client=args.num_classes_per_client)
    elif args.dataset == "JSRT":
        outdir = 'dataset/JSRT/'
        if ( args.dataset_outdir != None ):
            outdir = 'dataset/' + args.dataset_outdir + '/'
        generate_jsrt( args, outdir, args.num_clients, 100, args.dataset_niid, args.dataset_balance, args.dataset_partition, args.dataset_dir_alpha, class_per_client=args.num_classes_per_client)
    elif args.dataset == "Darwin":
        outdir = 'dataset/Darwin/'
        if ( args.dataset_outdir != None ):
            outdir = 'dataset/' + args.dataset_outdir + '/'
        generate_darwin( args, outdir, args.num_clients, 100, args.dataset_niid, args.dataset_balance, args.dataset_partition, args.dataset_dir_alpha, class_per_client=args.num_classes_per_client)
    elif args.dataset == "ShenzenMontgomery":
        outdir = 'dataset/ShenzhenMontgomerySeg/'
        if ( args.dataset_outdir != None ):
            outdir = 'dataset/' + args.dataset_outdir + '/'
        generate_shenzen_montgomery( args, outdir, args.num_clients, 100, args.dataset_niid, args.dataset_balance, args.dataset_partition, args.dataset_dir_alpha, class_per_client=args.num_classes_per_client)
    
    # else:
    #     generate_synthetic('dataset/synthetic/', args.num_clients, 10, args.niid)
    sys.exit(0)