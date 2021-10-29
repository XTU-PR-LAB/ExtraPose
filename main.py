import os
import sys
import time
import json
from pprint import pprint
from six.moves import xrange
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from src.opt import Options
from src.model import LinearModel,weight_init
import src.data_utils as data_utils
import src.cameras as cameras
import src.utils as utils
import src.procrustes as procrustes
import src.viz_new as viz


def get_all_batches(opt,data_x, data_y,training=True ):
    """
    Obtain a list of all the batches, randomly permutted
    Args
      data_x: dictionary with 2d inputs
      data_y: dictionary with 3d expected outputs
      camera_frame: whether the 3d data is in camera coordinates
      training: True if this is a training batch. False otherwise.

    Returns
      encoder_inputs: list of 2d batches
      decoder_outputs: list of 3d batches
    """

    # Figure out how many frames we have
    n = 0

    for key2d in data_x.keys():
      n2d, _ = data_x[ key2d ].shape
      n = n + n2d

    # 2d pos 具有 16个关节点
    encoder_inputs  = np.zeros((n, 16*2), dtype=float)
    # 3d pose 
    
    if opt.predict_14:
      decoder_outputs = np.zeros((n, 14*3), dtype=float)
    else: 
      decoder_outputs = np.zeros((n, 16*3),dtype=float)

    # Put all the data into big arrays
    idx = 0
    for key2d in data_x.keys():
      (subj, b, fname) = key2d
      # keys should be the same if 3d is in camera coordinates
      key3d = key2d if (opt.camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
      key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and opt.camera_frame else key3d

      n2d, _ = data_x[ key2d ].shape
      encoder_inputs[idx:idx+n2d, :]  = data_x[ key2d ]
      decoder_outputs[idx:idx+n2d, :] = data_y[ key3d ]
      idx = idx + n2d


    if training:
      # Randomly permute everything
      idx = np.random.permutation( n )
      encoder_inputs  = encoder_inputs[idx, :]
      decoder_outputs = decoder_outputs[idx, :]

    # Make the number of examples a multiple of the batch size
    n_extra  = n % opt.batch_size
    if n_extra > 0:  # Otherwise examples are already a multiple of batch size
      encoder_inputs  = encoder_inputs[:-n_extra, :]
      decoder_outputs = decoder_outputs[:-n_extra, :]

    n_batches = n // opt.batch_size
    encoder_inputs  = np.split( encoder_inputs, n_batches )
    decoder_outputs = np.split( decoder_outputs, n_batches )

    return encoder_inputs, decoder_outputs

def main(opt):
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr
    lr_decay = opt.lr_decay
    lr_init = opt.lr
    lr_gamma = opt.lr_gamma
    start_epoch = 0

    file_path = os.path.join(opt.ckpt, 'opt.json')
    with open(file_path, 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=True, indent=4))

    # create model
    print(">>> creating model")
    model = LinearModel(opt.batch_size,opt.predict_14)
    # = refine_2d_model(opt.batch_size,opt.predict_14)
    model = model.cuda()
    model.apply(weight_init)

    #refine_2d_model = refine_2d_model.cuda()
    #refine_2d_model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0) )#+ sum(p.numel() for p in refine_2d_model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    #refine_2d_model_optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # 加载checkpoint
    if opt.resume:
      print(">>> loading ckpt from '{}'".format(opt.load))
      ckpt = torch.load(opt.load)
      start_epoch = ckpt['epoch']
      err_best = ckpt['err']
      glob_step = ckpt['step']
      lr_now = ckpt['lr']
      model.load_state_dict(ckpt['state_dict'])
      #refine_2d_model.load_state_dict[ckpt['refine_state_dict']]
      optimizer.load_state_dict(ckpt['optimizer'])
      #refine_2d_model_optimizer.load_state_dict(ckpt['refine_optimizer'])
      print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # 包含动作的 list
    actions = data_utils.define_actions(opt.action)
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))
    pprint(actions, indent=4)
    print(">>>")

    # data loading
    print(">>> loading data")
  
    # Load camera parameters
    SUBJECT_IDS = [1,5,6,7,8,9,11]
    rcams = cameras.load_cameras(opt.cameras_path, SUBJECT_IDS)

    # Load 3d data and load (or create) 2d projections
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(actions, opt.data_dir, opt.camera_frame, rcams, opt.predict_14)
        
    # Read stacked hourglass 2D predictions if use_sh, otherwise use groundtruth 2D projections
    if opt.use_hg:
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, opt.data_dir)
    else: 
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, opt.data_dir, rcams )

    #gt_train_set_2d, gt_test_set_2d, gt_data_mean_2d, gt_data_std_2d, gt_dim_to_ignore_2d, gt_dim_to_use_2d = data_utils.create_2d_data( actions, opt.data_dir, rcams )

    print( "done reading and normalizing data." )

    step_time, loss = 0, 0
    current_epoch =  start_epoch
    log_every_n_batches = 100

    cudnn.benchmark = True
    best_error = 10000
    while current_epoch < opt.epochs:
        current_epoch = current_epoch + 1
       
        # === Load training batches for one epoch ===
        encoder_inputs, decoder_outputs = get_all_batches(opt, train_set_2d, train_set_3d, training=True )
        
        nbatches = len( encoder_inputs )
        print("There are {0} train batches".format( nbatches ))
        start_time = time.time()

        # === Loop through all the training batches ===
        current_step = 0
        for i in range( nbatches ):
            
            if (i+1) % log_every_n_batches == 0:
                # Print progress every log_every_n_batches batches
                print("Working on epoch {0}, batch {1} / {2}... \n".format( current_epoch, i+1, nbatches), end="" )

            model.train()

            if glob_step % lr_decay == 0 or glob_step == 1:
                lr_now = utils.lr_decay(optimizer, glob_step, lr_init, lr_decay, lr_gamma)
                #utils.lr_decay(refine_2d_model_optimizer, glob_step, lr_init, lr_decay, lr_gamma)

            enc_in = torch.from_numpy(encoder_inputs[i]).float()
            dec_out = torch.from_numpy(decoder_outputs[i]).float()
            
            inputs = Variable(enc_in.cuda())
            targets = Variable(dec_out.cuda())
            
            outputs = model(inputs)

            # calculate loss
            optimizer.zero_grad()
             
            step_loss = criterion(outputs, targets)
            step_loss.backward()

            if opt.max_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                #nn.utils.clip_grad_norm_(refine_2d_model.parameters(), max_norm=1)

            optimizer.step()

            loss += float(step_loss)

            current_step += 1
            glob_step += 1
            # === end looping through training batches ===

        
        loss = loss/nbatches
        
        print("=============================\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "=============================" % (glob_step, lr_now, loss) )
        # === End training for an epoch ===

         
        # clear useless chache
        torch.cuda.empty_cache()
        
        # === Testing after this epoch ===
        model.eval()
        if opt.evaluateActionWise:
          print("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs
          
          cum_err = 0
          record = ''
          for action in actions:

            print("{0:<12} ".format(action), end="")
            # Get 2d and 3d testing data for this action
            action_test_set_2d = get_action_subset( test_set_2d, action )
            action_test_set_3d = get_action_subset( test_set_3d, action )
            encoder_inputs, decoder_outputs = get_all_batches(opt, action_test_set_2d, action_test_set_3d ,training=False)

            total_err, joint_err, step_time = evaluate_batches( opt,criterion, model,
              data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
              data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
              current_step, encoder_inputs, decoder_outputs )
            cum_err = cum_err + total_err

            print("{0:>6.2f}".format(total_err))

            record = record + "{}   :   {}  (mm) \n".format(action, total_err)
          avg_val = cum_err/float(len(actions) )
          print("{0:<12} {1:>6.2f}".format("Average", avg_val))
          print("{0:=^19}".format(''))

          f = open("records.txt",'a')
          f.write("epoch: {} , avg_error: {}  loss : {} \n".format(current_epoch, avg_val,loss))

          if best_error > avg_val:
            print("=============================")
            print("==== save best record   =====")
            print("=============================")
            best_error = avg_val
            # save ckpt
            file_path = os.path.join(opt.ckpt, 'ckpt_last.pth.tar')
            torch.save({'epoch': current_epoch,
                    'lr': lr_now,
                    'step': glob_step,
                    'err': avg_val,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, file_path) 

            f.write("epoch: {} , avg_error: {} \n".format(current_epoch, best_error))
            f.write(record)
            
          f.write("=======================================\n")
          f.close()

          
        else:

          n_joints = 17 if not(opt.predict_14) else 14

          encoder_inputs, decoder_outputs = get_all_batches(opt,test_set_2d, test_set_3d, training=False)

          total_err, joint_err, step_time = evaluate_batches(opt, criterion,model,
            data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
            data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
            current_step, encoder_inputs, decoder_outputs, current_epoch )

          print("=============================\n"
              "Step-time (ms):      %.4f\n"
              "Val loss avg:        %.4f\n"
              "Val error avg (mm):  %.2f\n"
              "=============================" % ( 1000*step_time, loss, total_err ))

          
          for i in range(n_joints):
            # 6 spaces, right-aligned, 5 decimal places
            print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i+1, joint_err[i]))
            
            if save_flag is True:
              f.write("Error in joint {0:02d} (mm): {1:>5.2f} \n".format(i+1, joint_err[i]))  
          print("=============================")
          
          save_flag = False
          f.close()
          
    
    print( "done in {0:.2f} ms".format(1000*(time.time() - start_time)) ) 
    # Reset global time and loss
    step_time, loss = 0, 0


def evaluate_batches(opt,criterion,model,
  data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
  data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
  current_step, encoder_inputs, decoder_outputs, current_epoch=0 ):
  """
  Generic method that evaluates performance of a list of batches.
  May be used to evaluate all actions or a single action.

  Args
    sess
    model
    data_mean_3d
    data_std_3d
    dim_to_use_3d
    dim_to_ignore_3d
    data_mean_2d
    data_std_2d
    dim_to_use_2d
    dim_to_ignore_2d
    current_step
    encoder_inputs
    decoder_outputs
    current_epoch
  Returns

    total_err
    joint_err
    step_time
    loss
  """

  n_joints = 17 if not(opt.predict_14) else 14
  nbatches = len( encoder_inputs )

  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 100
  for i in range(nbatches):

    if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
      print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )
    
    enc_in = torch.from_numpy(encoder_inputs[i]).float()
    dec_out = torch.from_numpy(decoder_outputs[i]).float()

    

    inputs = Variable(enc_in.cuda())
    targets = Variable(dec_out.cuda())

    outputs = model(inputs)
    
    # denormalize
    enc_in  = data_utils.unNormalizeData( enc_in,  data_mean_2d, data_std_2d, dim_to_ignore_2d )
    dec_out = data_utils.unNormalizeData( dec_out, data_mean_3d, data_std_3d, dim_to_ignore_3d )
    poses3d = data_utils.unNormalizeData( outputs.data.cpu().numpy(), data_mean_3d, data_std_3d, dim_to_ignore_3d )

    # Keep only the relevant dimensions
    dtu3d = np.hstack( (np.arange(3), dim_to_use_3d) ) if not(opt.predict_14) else  dim_to_use_3d

    dec_out = dec_out[:, dtu3d]
    poses3d = poses3d[:, dtu3d]

    assert dec_out.shape[0] == opt.batch_size
    assert poses3d.shape[0] == opt.batch_size

    if opt.procrustes:
      # Apply per-frame procrustes alignment if asked to do so
      for j in range(opt.batch_size):
        gt  = np.reshape(dec_out[j,:],[-1,3])
        out = np.reshape(poses3d[j,:],[-1,3])
        _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
        out = (b*out.dot(T))+c

        poses3d[j,:] = np.reshape(out,[-1,17*3] ) if not(opt.predict_14) else np.reshape(out,[-1,14*3] )

    # Compute Euclidean distance error per joint
    sqerr = (poses3d - dec_out)**2 # Squared error between prediction and expected output
    dists = np.zeros( (sqerr.shape[0], n_joints) ) # Array with L2 error per joint in mm
    dist_idx = 0
    for k in np.arange(0, n_joints*3, 3):
      # Sum across X,Y, and Z dimenstions to obtain L2 distance
      dists[:,dist_idx] = np.sqrt( np.sum( sqerr[:, k:k+3], axis=1 ))
      dist_idx = dist_idx + 1

    all_dists.append(dists)
    assert sqerr.shape[0] == opt.batch_size

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  all_dists = np.vstack( all_dists )

  # Error per joint and total for all passed batches
  joint_err = np.mean( all_dists, axis=0 )
  total_err = np.mean( all_dists )

  return total_err, joint_err, step_time


def get_action_subset( poses_set, action ):
  """
  Given a preloaded dictionary of poses, load the subset of a particular action

  Args
    poses_set: dictionary with keys k=(subject, action, seqname),
      values v=(nxd matrix of poses)
    action: string. The action that we want to filter out
  Returns
    poses_subset: dictionary with same structure as poses_set, but only with the
      specified action.
  """
  return {k:v for k, v in poses_set.items() if k[1] == action}

def sample(opt):
  """Get samples from a model and visualize them"""

  actions = data_utils.define_actions( opt.action )

  # Load camera parameters
  SUBJECT_IDS = [1,5,6,7,8,9,11]
  rootPath = os.getcwd()
  rcams = cameras.load_cameras(os.path.join(rootPath, opt.cameras_path), SUBJECT_IDS)

  # Load 3d data and load (or create) 2d projections
  train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
    actions, opt.data_dir, opt.camera_frame, rcams, opt.predict_14 )

  if opt.use_hg:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, opt.data_dir)
  else:
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, opt.data_dir, rcams )
  print( "done reading and normalizing data." )

  # create model
  print(">>> creating model")
  model = LinearModel(opt.batch_size,opt.predict_14)
  model = model.cuda()
  model.apply(weight_init)

  optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

  print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

  print(">>> loading ckpt from '{}'".format(opt.load))

  ckpt = torch.load(opt.load)
  model.load_state_dict(ckpt['state_dict'])
  optimizer.load_state_dict(ckpt['optimizer'])
  print("Model loaded")
  model.eval()


  for key2d in test_set_2d.keys():

    (subj, b, fname) = key2d
    print( "Subject: {}, action: {}, fname: {}".format(subj, b, fname) )

    # keys should be the same if 3d is in camera coordinates
    key3d = key2d if opt.camera_frame else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
    key3d = (subj, b, fname[:-3]) if (fname.endswith('-sh')) and opt.camera_frame else key3d

    enc_in  = test_set_2d[ key2d ]
    n2d, _ = enc_in.shape
    dec_out = test_set_3d[ key3d ]
    n3d, _ = dec_out.shape
    assert n2d == n3d

      # Split into about-same-size batches
    enc_in   = np.array_split( enc_in,  n2d // opt.batch_size )
    dec_out  = np.array_split( dec_out, n3d // opt.batch_size )
    all_poses_3d = []

    for bidx in range( len(enc_in) ):

      # Dropout probability 0 (keep probability 1) for sampling
      dp = 1.0
      ei = torch.from_numpy(enc_in[bidx]).float()
      inputs = Variable(ei.cuda())
      outputs = model(inputs)
      
      # denormalize
      enc_in[bidx]  = data_utils.unNormalizeData(  enc_in[bidx], data_mean_2d, data_std_2d, dim_to_ignore_2d )
      dec_out[bidx] = data_utils.unNormalizeData( dec_out[bidx], data_mean_3d, data_std_3d, dim_to_ignore_3d )
      poses3d = data_utils.unNormalizeData(  outputs.data.cpu().numpy(), data_mean_3d, data_std_3d, dim_to_ignore_3d )
      all_poses_3d.append( poses3d )

    # Put all the poses together
    enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, all_poses_3d] )

    # Convert back to world coordinates
    if opt.camera_frame:
      N_CAMERAS = 4
      N_JOINTS_H36M = 32

      # Add global position back
      dec_out = dec_out + np.tile( test_root_positions[ key3d ], [1,N_JOINTS_H36M] )

      # Load the appropriate camera
      subj, _, sname = key3d

      cname = sname.split('.')[1] # <-- camera name
      scams = {(subj,c+1): rcams[(subj,c+1)] for c in range(N_CAMERAS)} # cams of this subject
      scam_idx = [scams[(subj,c+1)][-1] for c in range(N_CAMERAS)].index( cname ) # index of camera used
      the_cam  = scams[(subj, scam_idx+1)] # <-- the camera used
      R, T, f, c, k, p, name = the_cam
      assert name == cname

      def cam2world_centered(data_3d_camframe):
        data_3d_worldframe = cameras.camera_to_world_frame(data_3d_camframe.reshape((-1, 3)), R, T)
        data_3d_worldframe = data_3d_worldframe.reshape((-1, N_JOINTS_H36M*3))
        # subtract root translation
        return data_3d_worldframe - np.tile( data_3d_worldframe[:,:3], (1,N_JOINTS_H36M) )

        # Apply inverse rotation and translation
      dec_out = cam2world_centered(dec_out)
      poses3d = cam2world_centered(poses3d)

  # Grab a random batch to visualize
  enc_in, dec_out, poses3d = map( np.vstack, [enc_in, dec_out, poses3d] )
  idx = np.random.permutation( enc_in.shape[0] )
  enc_in, dec_out, poses3d = enc_in[idx, :], dec_out[idx, :], poses3d[idx, :]

  # Visualize random samples
  import matplotlib.gridspec as gridspec

  # 1080p	= 1,920 x 1,080
  fig = plt.figure( figsize=(19.2, 10.8) )

  gs1 = gridspec.GridSpec(5, 9) # 5 rows, 9 columns
  gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
  plt.axis('off')

  subplot_idx, exidx = 1, 1
  nsamples = 15
  for i in np.arange( nsamples ):

    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    p2d = enc_in[exidx,:]
    viz.show2Dpose( p2d, ax1 )
    ax1.invert_yaxis()

    # Plot 3d gt
    ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
    p3d = dec_out[exidx,:]
    viz.show3Dpose( p3d, ax2 )

    # Plot 3d predictions
    ax3 = plt.subplot(gs1[subplot_idx+1], projection='3d')
    p3d = poses3d[exidx,:]
    viz.show3Dpose( p3d, ax3, lcolor="#9b59b6", rcolor="#2ecc71" )

    exidx = exidx + 1
    subplot_idx = subplot_idx + 3

  plt.show()


def get_bone_length_loss(output,target):
    raw = output.shape[0]
    output = output.reshape([raw,-1,3])
    target = target.reshape([raw,-1,3])

    parent = [0,0,1,2,0,4,5,0,7,8,8,10,11,8,13,14]

    dists1 = output - output[:,parent,:]
    dists2 = target - target[:,parent,:]

    output_boneLengths = torch.mean(torch.norm(dists1, dim=2), dim=1)
    target_boneLengths = torch.mean(torch.norm(dists2, dim=2), dim=1)

    penalty = torch.mean(torch.abs(output_boneLengths - target_boneLengths))
    return penalty



def testFunc(opt):
    start_epoch = 0
    print("procrustes          {}".format(opt.procrustes))
    # create model
    print(">>> creating model")
    model = LinearModel(opt.batch_size,opt.predict_14)
    # = refine_2d_model(opt.batch_size,opt.predict_14)
    model = model.cuda()
    model.apply(weight_init)
    model.eval()
    #refine_2d_model = refine_2d_model.cuda()
    #refine_2d_model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0) )#+ sum(p.numel() for p in refine_2d_model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    #refine_2d_model_optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # 加载checkpoint
    print(">>> loading ckpt from '{}'".format(opt.load))
    ckpt = torch.load(opt.load)
    start_epoch = ckpt['epoch']
    err_best = ckpt['err']
    glob_step = ckpt['step']
    model.load_state_dict(ckpt['state_dict'])
      #refine_2d_model.load_state_dict[ckpt['refine_state_dict']]
    optimizer.load_state_dict(ckpt['optimizer'])
      #refine_2d_model_optimizer.load_state_dict(ckpt['refine_optimizer'])
    print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # 包含动作的 list
    actions = data_utils.define_actions(opt.action)
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))
    pprint(actions, indent=4)
    print(">>>")

    # data loading
    print(">>> loading data")
  
    # Load camera parameters
    SUBJECT_IDS = [1,5,6,7,8,9,11]
    rcams = cameras.load_cameras(opt.cameras_path, SUBJECT_IDS)

    # Load 3d data and load (or create) 2d projections
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(actions, opt.data_dir, opt.camera_frame, rcams, opt.predict_14)
        
    # Read stacked hourglass 2D predictions if use_sh, otherwise use groundtruth 2D projections
    if opt.use_hg:
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, opt.data_dir)
    else:
        train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.create_2d_data( actions, opt.data_dir, rcams )

    #gt_train_set_2d, gt_test_set_2d, gt_data_mean_2d, gt_data_std_2d, gt_dim_to_ignore_2d, gt_dim_to_use_2d = data_utils.create_2d_data( actions, opt.data_dir, rcams )

    print( "done reading and normalizing data." )

    cudnn.benchmark = True
       
        # === Testing after this epoch ===
    if opt.evaluateActionWise:
        print("{0:=^12} {1:=^6}".format("Action", "mm")) # line of 30 equal signs
          
        cum_err = 0
        record = ''
        for action in actions:

            print("{0:<12} ".format(action), end="")
            # Get 2d and 3d testing data for this action
            action_test_set_2d = get_action_subset( test_set_2d, action )
            action_test_set_3d = get_action_subset( test_set_3d, action )
            encoder_inputs, decoder_outputs = get_all_batches(opt, action_test_set_2d, action_test_set_3d, rcams)

            total_err, joint_err, step_time = evaluate_batches( opt,criterion, model,
              data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
              data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
              glob_step, encoder_inputs, decoder_outputs )
            cum_err = cum_err + total_err

            print("{0:>6.2f}".format(total_err))

            record = record + "{}   :   {}  (mm) \n".format(action, total_err)
        avg_val = cum_err/float(len(actions) )
        print("{0:<12} {1:>6.2f}".format("Average", avg_val))
        print("{0:=^19}".format(''))

        f = open(opt.ckpt + "records.txt",'a')
        f.write("Test --- epoch: {} , avg_error: {}  loss : {} \n".format(start_epoch, avg_val,err_best))
        f.write(record)
        f.write("=======================================\n")
        f.close()

          
    else:

        n_joints = 17 if not(opt.predict_14) else 14

        encoder_inputs, decoder_outputs = get_all_batches(opt,test_set_2d, test_set_3d, rcams, training=False)

        total_err, joint_err, step_time = evaluate_batches(opt, criterion,model,
            data_mean_3d, data_std_3d, dim_to_use_3d, dim_to_ignore_3d,
            data_mean_2d, data_std_2d, dim_to_use_2d, dim_to_ignore_2d,
            glob_step, encoder_inputs, decoder_outputs, start_epoch )

        print("=============================\n"
              "Step-time (ms):      %.4f\n"
              "Val loss avg:        %.4f\n"
              "Val error avg (mm):  %.2f\n"
              "=============================" % ( 1000*step_time, loss, total_err ))

          
        for i in range(n_joints):
            # 6 spaces, right-aligned, 5 decimal places
            print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i+1, joint_err[i]))
            
            if save_flag is True:
              f.write("Error in joint {0:02d} (mm): {1:>5.2f} \n".format(i+1, joint_err[i]))  
            print("=============================")
          
        save_flag = False
        f.close()
          



if __name__ == "__main__":
    option = Options().parse()
    if option.sample:
      sample(option)
    elif option.test:
      testFunc(option)
    else:
      main(option)
