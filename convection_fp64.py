import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS,Adam
from tqdm import tqdm
import os
import argparse
from util import *
from model_dict import get_model

#torch.set_float64_matmul_precision('high')
#torch.backends.cuda.matmul.allow_tf32 = False

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
step_size = 1e-4
num_step=5
beta=50

parser = argparse.ArgumentParser('Training Point Optimization')
parser.add_argument('--model', type=str, default='pinn')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
device = args.device

res, b_left, b_right, b_upper, b_lower = get_data([0, 2 * np.pi], [0, 1], 401, 401)
res_test, b_left_test, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)

if args.model == 'PINNsFormer' or args.model == 'PINNMamba':
    res = make_time_sequence(res, num_step=num_step, step=step_size)
    b_left = make_time_sequence(b_left, num_step=num_step, step=step_size)
    b_right = make_time_sequence(b_right, num_step=num_step, step=step_size)
    b_upper = make_time_sequence(b_upper, num_step=num_step, step=step_size)
    b_lower = make_time_sequence(b_lower, num_step=num_step, step=step_size)
    
if args.model == 'PINNsFormer' or args.model == 'PINNMamba':
    res_test = make_time_sequence(res_test, num_step=num_step, step=step_size)

res = torch.tensor(res, dtype=torch.float64, requires_grad=True).to(device)
b_left = torch.tensor(b_left, dtype=torch.float64, requires_grad=True).to(device)
b_right = torch.tensor(b_right, dtype=torch.float64, requires_grad=True).to(device)
b_upper = torch.tensor(b_upper, dtype=torch.float64, requires_grad=True).to(device)
b_lower = torch.tensor(b_lower, dtype=torch.float64, requires_grad=True).to(device)

x_res, t_res = res[:, ..., 0:1], res[:, ..., 1:2]
x_left, t_left = b_left[:, ..., 0:1], b_left[:, ..., 1:2]
x_right, t_right = b_right[:, ..., 0:1], b_right[:, ..., 1:2]
x_upper, t_upper = b_upper[:, ..., 0:1], b_upper[:, ..., 1:2]
x_lower, t_lower = b_lower[:, ..., 0:1], b_lower[:, ..., 1:2]

print(t_res)
def init_weights(m):
    if isinstance(m, nn.Linear):
        #if(m.bias):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)


if args.model == 'KAN':
    model = get_model(args).Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, \
                                  noise_scale_base=0.25, device=device).to(torch.float64).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(torch.float64).to(device)
    model.apply(init_weights)
elif args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
    model = get_model(args).Model(in_dim=2, hidden_dim=32, out_dim=1, num_layer=1).to(torch.float64).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(torch.float64).to(device)
    model.apply(init_weights)



optim = Adam(model.parameters(),lr=1e-6)
#optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe', tolerance_grad = 1e-8, tolerance_change= 1e-10)
print(model)
print(get_n_params(model))

loss_track = []

gradient_stats = []

print(model.named_parameters())

for i in tqdm(range(50000)):
    def closure():
        pred_res = model(x_res, t_res)
        pred_left = model(x_left, t_left)
        pred_right = model(x_right, t_right)
        pred_upper = model(x_upper, t_upper)
        pred_lower = model(x_lower, t_lower)

        u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                  create_graph=True)[0]
        u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                  create_graph=True)[0]

        loss_res = torch.mean((u_t + beta * u_x) ** 2)
        loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
        loss_ic = torch.mean((pred_left[:,0] - torch.sin(x_left[:,0])) ** 2)

        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])
        #print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
        loss = loss_res + loss_bc + loss_ic
        #loss = 1000*loss
        #loss = loss_ic
        optim.zero_grad()
        loss.backward()



        
    #
        return loss


    optim.step(closure)

    grad_norms = []
    grad_means = []
    grad_stds = []


    if i>50:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()  
                    grad_mean = param.grad.mean().item()       
                    grad_std = param.grad.std().item()      
                    grad_norms.append(grad_norm)
                    grad_means.append(grad_mean)
                    grad_stds.append(grad_std)

           
            gradient_stats.append({
                'step': i,
                'grad_norms': grad_norms,
                'grad_means': grad_means,
                'grad_stds': grad_stds
            })


            print(f"Step {i}:")
            print(f"  Gradient Norms: {grad_norms}")
            print(f"  Gradient Means: {grad_means}")
            print(f"  Gradient Stds: {grad_stds}")



print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))

if not os.path.exists('./results/'):
    os.makedirs('./results/')

torch.save(model.state_dict(), f'./results/1dconvection_{args.model}_{num_step}_{step_size}.pt')

# Visualize


res_test = torch.tensor(res_test, dtype=torch.float64, requires_grad=True).to(device)
x_test, t_test = res_test[:, ..., 0:1], res_test[:, ..., 1:2]
x_left_test, t_left_test = b_left_test[:, ..., 0:1], b_left_test[:, ..., 1:2]


#print(t_test)
#with torch.no_grad():
pred = model(x_test, t_test)[:, 0:1]
u_x = torch.autograd.grad(pred, x_test, grad_outputs=torch.ones_like(pred), retain_graph=True,
                                  create_graph=True)[0]
u_t = torch.autograd.grad(pred, t_test, grad_outputs=torch.ones_like(pred), retain_graph=True,
                                  create_graph=True)[0]
#loss_res = (u_t + 50 * u_x) ** 2


    #print(pred.shape)
    #pred = model(x_test, t_test)[:, 0:1]
#loss_res = loss_res.cpu().detach().numpy()
#loss_res = loss_res.reshape(101, 101)

#print(loss_res)

pred = pred.cpu().detach().numpy()

#print(pred.shape) 
pred = pred.reshape(101, 101)


def u_res(x, t):
    #print(x.shape)
    #print(t.shape)
    return np.sin(x - beta * t)


res_test, _, _, _, _ = get_data([0, 2 * np.pi], [0, 1], 101, 101)
u = u_res(res_test[:, 0], res_test[:, 1]).reshape(101, 101)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

print(beta)
print('relative L1 error: {:4f}'.format(rl1))
print('relative L2 error: {:4f}'.format(rl2))

plt.figure(figsize=(4, 3))
plt.imshow(pred,extent=[0,np.pi*2,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted u(x,t)')
plt.colorbar()
plt.tight_layout()
#plt.axis('off')
plt.savefig(f'./results/convection_{args.model}_{num_step}_{step_size}_{beta}_pred.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(u,extent=[0,np.pi*2,1,0], aspect='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
#plt.axis('off')
plt.savefig('./results/convection_exact_{beta}.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(pred - u, extent=[0,np.pi*2,1,0], aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute Error')
plt.colorbar()
plt.tight_layout()
#plt.axis('off')
plt.savefig(f'./results/convection_{args.model}_{num_step}_{step_size}_{beta}_error.pdf', bbox_inches='tight')

grad_norms_history = [stats['grad_norms'] for stats in gradient_stats]
grad_means_history = [stats['grad_means'] for stats in gradient_stats]
grad_stds_history = [stats['grad_stds'] for stats in gradient_stats]


plt.figure(figsize=(10, 6))
for i in range(len(grad_norms_history[0])):
    plt.plot([stats[i] for stats in grad_norms_history], label=f'Layer {i+1}')
plt.xlabel('Training Step')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms During Training')
plt.legend()
plt.grid()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_fp64_gradient_norms.pdf')  
plt.close()  


plt.figure(figsize=(10, 6))
for i in range(len(grad_means_history[0])):
    plt.plot([stats[i] for stats in grad_means_history], label=f'Layer {i+1}')
plt.xlabel('Training Step')
plt.ylabel('Gradient Mean')
plt.title('Gradient Means During Training')
plt.legend()
plt.grid()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_fp64_gradient_means.pdf')  
plt.close()  


plt.figure(figsize=(10, 6))
for i in range(len(grad_stds_history[0])):
    plt.plot([stats[i] for stats in grad_stds_history], label=f'Layer {i+1}')
plt.xlabel('Training Step')
plt.ylabel('Gradient Std')
plt.title('Gradient Stds During Training')
plt.legend()
plt.grid()
plt.savefig(f'./results/1d_convection_{args.model}_{beta}_fp64_gradient_stds.pdf')  
plt.close()  
