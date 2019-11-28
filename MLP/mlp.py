import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import SGD, Adam
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import os



use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data)
        self.target = torch.from_numpy(target)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--n_iters',
                    required=True,
                    type=int,
                    default=100,
                    help='Number of iterations to train')

    p.add_argument('--lr',
                    required=True,
                    type=float,
                    help='learning rate')

    p.add_argument('--input_size',
                    type=int,
                    default=1024,
                    help='the dimension of our input vectors')

    p.add_argument('--hidden_size',
                    type=int,
                    default= 1024,
                    help='Hidden size')

    p.add_argument('--batch_size',
                    type=int,
                    default= 32,
                    help='batch size')

    p.add_argument('--output_size',
                    type=int,
                    default= 1024,
                    help='Output size')
    
    p.add_argument('--dropout_p',
                    type=float,
                    default=.1,
                    help='Dropout ratio. Default=.1')

    p.add_argument('--checkpoint_path',
                    type=str,
                    default= '/',
                    help='path to save checkpoints during training')

    config = p.parse_args()
    return config

def combine_reshape_laser_array(simple1, simple2):
    simple_sent = []
    for i in range(simple1.shape[0]):
        arr = np.concatenate((simple1[i], simple2[i]), axis=0)
        #arr = arr.reshape(1, -1)
        simple_sent.append(arr)
    simple_sent = np.array(simple_sent)
    return simple_sent
class MLPNet(nn.Module):
    def __init__(self, args):
        super(MLPNet, self).__init__()
        self.input_dim = args.input_size
        self.hidden_dim = args.hidden_size
        self.output_dim = args.output_size
        self.dropout_prob = args.dropout_p

        self.fc1 = nn.Linear(2*self.input_dim, self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # #self.batch_norm = nn.BatchNorm1d(self.hidden_dim)

        self.out_layer = nn.Linear(self.hidden_dim, self.output_dim)
        # #self.fc3 = nn.Linear(output_dim, output_dim)
        # self.tanh = nn.Tanh()

    def forward(self, inputs):
        input_vec = self.fc1(inputs)
        input_vec = F.tanh(input_vec)

        input_vec = self.fc2(input_vec)
        # #input_vec = self.batch_norm(input_vec)
        input_vec = F.tanh(input_vec)
        #input_vec = F.dropout(input_vec, self.dropout_prob)
        
        input_vec = self.out_layer(input_vec)
        #input_vec = self.tanh(input_vec)
        return input_vec #torch.mean(input_vec, dim=input_vec.size(2))

def find_el(arr, el):
    i = 0
    while i < len(arr) and arr[i] != el:
        i+= 1
    if i < len(arr):
        return i
    else:
        return -1

def train_mlp(model, train_dataloader,valid_dataloader, criterion, optimizer,  args):
    def save_checkpoint(args, epoch, name):
        checkpoint = {
                'model': MLPNet(args),
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }
        torch.save(checkpoint, args.checkpoint_path+'/'+name)
        print('saved checkpoint at {}'.format(name))
    losses = []
    accuracies = []
    prev_max = 0
    f = open('ranking_mlp_roberta', 'w')
    for epoch in range(args.n_iters):  # loop over the dataset multiple times
        model.train()
        training_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels
            #print(data)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            data = data.float()
            labels = labels.float()

            if torch.cuda.is_available():
                model = model.cuda()
                data = Variable(data.cuda())
                labels = Variable(labels.cuda())
            else:
                data = Variable(data)
                labels = Variable(labels)
            outputs = model(data)
            loss = criterion(outputs, labels)
            #loss = loss.mean()
            loss.backward(retain_graph=True)
            optimizer.step()
            training_loss += loss.item()
            if batch_idx % 100 == 0:
                print ('Epoch: %04d/%04d | Batch %03d/%03d | Loss: %.7f' 
                    %(epoch+1, args.n_iters, batch_idx, 
                        len(train_dataloader), loss.item()))
        print ('Epoch: %04d/%04d  | Training Loss: %.7f' 
                    %(epoch+1, args.n_iters, training_loss/len(train_dataloader)))
        losses.append(training_loss/len(train_dataloader))
        with torch.no_grad():
            model.eval()
            accuracy = 0
            for batch_idx, (data, labels) in enumerate(valid_dataloader):
                data = data.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                if torch.cuda.is_available():
                    data = Variable(data.cuda())
                    labels = Variable(labels.cuda())
                else:
                    data = Variable(labels)
                    labels = Variable(labels)
                outputs = model(data)
                loss = []
                for k in range(labels.size(1)):
                    label = labels[:,k,:]
                    loss.append(criterion(outputs, label).item())
                loss_copy = loss.copy()
                if epoch == args.n_iters - 1:
                    loss1 = np.array(loss)
                    np.savetxt(f, loss1)
                # loss_copy.sort()
                # ind_true = find_el(loss_copy, loss[0])
                # accuracy += 1/(ind_true + 1)
                min_loss = np.argmin(loss)
                if min_loss == 0:
                    accuracy += 1
            print ('Epoch: %04d/%04d  | Validation accuracy: %.7f' 
                    %(epoch+1, args.n_iters, accuracy/len(valid_dataloader)))
        accuracies.append(accuracy/len(valid_dataloader)) 
        if epoch == 0:
            save_checkpoint(args, epoch, 'checkpoint_best.pt')
            save_checkpoint(args, epoch, 'checkpoint_last.pt')
            prev_max = np.max(accuracies)
        else:
            max_acc = np.max(accuracies)
            if max_acc > prev_max:
                save_checkpoint(args, epoch, 'checkpoint_best.pt')
                prev_max = max_acc
            os.rename(args.checkpoint_path+'/checkpoint_last.pt', args.checkpoint_path+'/checkpoint'+str(epoch-1)+'.pt')
            save_checkpoint(args, epoch, 'checkpoint_last.pt')
        
        if epoch % 100 == 0 and epoch > 0:
            losse_array = np.array(losses)
            accuracies_array = np.array(accuracies)
            np.savetxt('results/laser/losses_'+str(epoch)+'.csv', losse_array)
            np.savetxt('results/laser/accuracies_'+str(epoch)+'.csv', accuracies_array)
    losses = np.array(losses)
    accuracies = np.array(accuracies)
    np.savetxt('losses1.csv', losses)
    np.savetxt('accuracies1.csv', accuracies)
    print('Finished Training')
    print('maximum validation accuracy: {}'.format(np.max(accuracies)))
    f.close()
def main():    
    args = argparser()
    input_dim = args.input_size
    hidden_dim = args.hidden_size
    output_dim = args.output_size
    dropout_p = args.dropout_p
    n_iters = args.n_iters
    # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
    # roberta.eval()
    mlpNet = MLPNet(args)
    criterion = nn.MSELoss()
    optimizer = Adam(mlpNet.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)

    ##load data
    
    #load training data
    dim = 1024
    numpy_train_simple1 = np.fromfile('training_data/training_simple1.raw', dtype=np.float32, count=-1)                                                                        
    numpy_train_simple1.resize(numpy_train_simple1.shape[0] // dim, dim )
    print('finish loading training data simple 1 of shape: ', numpy_train_simple1.shape)
    numpy_train_simple2 = np.fromfile('training_data/training_simple2.raw', dtype=np.float32, count=-1)                                                                        
    numpy_train_simple2.resize(numpy_train_simple2.shape[0] // dim, dim )
    print('finish loading training data simple 2 of shape: ', numpy_train_simple2.shape)

    numpy_train_data = combine_reshape_laser_array(numpy_train_simple1, numpy_train_simple2)
    print('finish loading training data of shape: ', numpy_train_data.shape)
    
    numpy_train_target = np.fromfile('training_data/training_complex_sent.raw' , dtype=np.float32, count=-1)
    numpy_train_target.resize(numpy_train_target.shape[0] // dim, dim )
    print('finish loading training targets of shape: ', numpy_train_target.shape)

    

    
    
    
    trainset = MyDataset(numpy_train_data, numpy_train_target)
    trainloader = DataLoader(
            trainset,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
    )

    #load validation data
    # numpy_val_data = np.loadtxt('LASER/my_data/data/laser_vectors/valid_simple')
    # numpy_val_data.resize(numpy_val_data.shape[0] // dim, dim)
    
    numpy_val_simple1 = np.fromfile("../../Laser/Data/valid_data/simple1.raw", dtype=np.float32, count=-1)
    numpy_val_simple1.resize(numpy_val_simple1.shape[0] // dim, dim )
    numpy_val_simple2 = np.fromfile("../../Laser/Data/valid_data/simple2.raw", dtype=np.float32, count=-1)
    numpy_val_simple2.resize(numpy_val_simple2.shape[0] // dim, dim)
    numpy_val_data = combine_reshape_laser_array(numpy_val_simple1, numpy_val_simple2)
    print("Validation data shape: ", numpy_val_data.shape)
    numpy_val_target = np.fromfile("../../Laser/Data/valid_data/paraphrase_sampled.raw", dtype=np.float32, count=-1)
    numpy_val_target.resize(numpy_val_target.shape[0]// dim, dim )
    numpy_val_target.resize(numpy_val_target.shape[0]//6, 6, numpy_val_target.shape[1])
    print("Validation target shape: ",  numpy_val_target.shape)
   

    validset = MyDataset(numpy_val_data, numpy_val_target)
    validloader = DataLoader(
            validset,
            batch_size=1,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
    )
    # print("training with Laser hits 100")
    train_mlp(mlpNet, trainloader, validloader, criterion, optimizer, args)

if __name__ == "__main__":
    main()
    # s = np.loadtxt('data/train_data.csv')
    # print(s.shape)

    
        