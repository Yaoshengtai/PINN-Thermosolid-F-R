m=50
ut_mat = torch.zeros(m,m)    
for i in range(m):
    for j in range(i, m):
        ut_mat[i, j] = 1


m=int(math.sqrt(uu.shape[0]))
if self.args.disp_accumulate==1 and m*m==self.args.batch_size:
            delta_r=(r2-r1)/m
            delta_z=h1/m
            e_r=(calculate_sigma_rr(uu[:,0].clone(),uu[:,1].clone(),xx,yy)- \
                mu*(calculate_sigma_theta(uu[:,0].clone(),uu[:,1].clone(),xx,yy)+calculate_sigma_zz(uu[:,0].clone(),uu[:,1].clone(),xx,yy)))/E *10**12
            e_z=(calculate_sigma_zz(uu[:,0].clone(),uu[:,1].clone(),xx,yy)- \
                mu*(calculate_sigma_theta(uu[:,0].clone(),uu[:,1].clone(),xx,yy)+calculate_sigma_rr(uu[:,0].clone(),uu[:,1].clone(),xx,yy)))/E *10**12
            uu[:,0]=torch.flatten(torch.mm(e_r.reshape(m,m),delta_r*ut_mat)).requires_grad_() 
            uu[:,1]=torch.flatten(torch.mm(e_z.reshape(m,m),delta_z*ut_mat)).requires_grad_() 
        