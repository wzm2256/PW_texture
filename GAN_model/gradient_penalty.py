import torch
from torch.autograd import grad

class Grad_Penalty:

    def __init__(self, lambdaGP, point_mass, gamma=1, device=torch.device('cpu'), ):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.device = device
        self.point_mass = point_mass

    def __call__(self, loss, All_points):

        gradients = grad(outputs=loss, inputs=[i.contiguous() for i in All_points], grad_outputs=torch.ones(loss.size()).to(self.device).contiguous(),
                    create_graph=True, retain_graph=True)

        g_square1 = []
        for g in gradients[:len(gradients) // 2]:
            Scale = gradients[0].shape[2] // g.shape[2]
            g_square1.append(
                torch.repeat_interleave(torch.repeat_interleave((g / self.point_mass[1]).norm(2, dim=[0, 1]), Scale, 0), Scale, 1))
        grad_norm1 = torch.stack(g_square1, 0).sum(0)
        gradient_penalty1 = (torch.nn.functional.relu(grad_norm1 - self.gamma)).mean() * self.lambdaGP

        g_square2 = []
        for g in gradients[len(gradients) // 2:]:
            Scale = gradients[len(gradients) // 2].shape[2] // g.shape[2]
            g_square2.append(
                torch.repeat_interleave(torch.repeat_interleave((g / self.point_mass[1]).norm(2, dim=[0, 1]), Scale, 0), Scale, 1))
        grad_norm2 = torch.stack(g_square2, 0).sum(0)
        gradient_penalty2 = (torch.nn.functional.relu(grad_norm2 - self.gamma)).mean() * self.lambdaGP

        with torch.no_grad():
            M1 = torch.max(grad_norm1)
            M2 = torch.max(grad_norm2)
            M = torch.max(torch.stack([M1, M2]))

        gradient_penalty = gradient_penalty1 + gradient_penalty2
        
        return gradient_penalty, M, grad_norm1, grad_norm2



