import numpy as np
from sklearn.model_selection import train_test_split


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import math
import torchvision as tv

from time import time
from src.model.madry_model import WideResNet
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from src.argument import parser, print_args

def __balance_val_split(dataset_da, dataset, val_split=2000):
    targets = np.array(dataset.targets)
    targets_da = np.array(dataset_da.targets)
    print(np.sum(targets==targets_da))
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets
    )
    train_dataset = Subset(dataset_da, indices=train_indices)
    val_dataset = Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset

class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        opt = torch.optim.SGD(model.parameters(), args.learning_rate, 
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)
        iter_per_epoch = math.ceil(48000.0/args.batch_size)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[iter_per_epoch*60, iter_per_epoch*120], 
                                                         gamma=0.1)
        _iter = 0

        begin_time = time()
        logit_grad_norm_decay = args.lambda_EMR
        print(model.conv1.weight.grad)
        criterion_kl = nn.KLDivLoss(size_average=False)
        best_va_adv_acc = 0.0
        for epoch in range(1, args.max_epoch+1):
            if epoch == 60 or epoch == 120:
                logit_grad_norm_decay = args.EMR_decay*logit_grad_norm_decay
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                adv_data = self.attack.perturb(data, label, 'mean', True)
                output = model(data, _eval=False)
                loss = F.cross_entropy(output, label)
                loss_robust = (1.0 / args.batch_size) * criterion_kl(F.log_softmax(model(adv_data), dim=1),
                                                    F.softmax(model(data), dim=1))


                opt.zero_grad()
                
                
                output_eval = model(adv_data, _eval=True)
                
                
                prob_eval = F.softmax(output_eval/args.EMR_softmax_temp,dim=1).detach()
                prob_logit = torch.mean(torch.sum(output_eval * prob_eval, dim=1))
                
                logit_grad = torch.autograd.grad(prob_logit,adv_data,create_graph=True)[0]
                
                #print(logit_grad.size())
                logit_grad_l2_norm = torch.sum(torch.square(logit_grad))
                
                #print(logit_grad_l2_norm)
#                 loss.backward()
                (logit_grad_norm_decay*logit_grad_l2_norm + loss + args.beta_trades * loss_robust).backward()
                print(logit_grad_norm_decay*logit_grad_l2_norm+loss)
                print(logit_grad_norm_decay)
                opt.step()

                if _iter % args.n_eval_step == 0:
                    t1 = time()

                    if adv_train:
                        with torch.no_grad():
                            stand_output = model(data, _eval=True)
                        pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    else:
                        
                        adv_data = self.attack.perturb(data, label, 'mean', False)

                        with torch.no_grad():
                            adv_output = model(adv_data, _eval=True)
                        pred = torch.max(adv_output, dim=1)[1]
                        # print(label)
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    t2 = time()

                    logger.info(f'epoch: {epoch}, iter: {_iter}, lr={opt.param_groups[0]["lr"]}, '
                                f'spent {time()-begin_time:.2f} s, tr_loss: {loss.item():.3f}')

                    logger.info(f'standard acc: {std_acc:.3f}%, robustness acc: {adv_acc:.3f}%')

                    # begin_time = time()

                    # if va_loader is not None:
                    #     va_acc, va_adv_acc = self.test(model, va_loader, True)
                    #     va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    #     logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                    #     logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    #         va_acc, va_adv_acc, time() - begin_time))
                    #     logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

                    begin_time = time()

                if _iter % args.n_store_image_step == 0:
                    tv.utils.save_image(torch.cat([data.cpu(), adv_data.cpu()], dim=0), 
                                        os.path.join(args.log_folder, f'images_{_iter}.jpg'), 
                                        nrow=16)

                #if _iter % args.n_checkpoint_step == 0:
                #    file_name = os.path.join(args.model_folder, f'checkpoint_{_iter}.pth')
                #    save_model(model, file_name)

                _iter += 1
                # scheduler depends on training interation
                scheduler.step()

            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc = self.test(model, va_loader, True, False)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time()
                logger.info('\n'+'='*20 +f' evaluation at epoch: {epoch} iteration: {_iter} ' \
                    +'='*20)
                logger.info(f'test acc: {va_acc:.3f}%, test adv acc: {va_adv_acc:.3f}%, spent: {t2-t1:.3f} s')
                logger.info('='*28+' end of evaluation '+'='*28+'\n')
                if va_adv_acc > best_va_adv_acc:
                    best_va_adv_acc = va_adv_acc
                    file_name = os.path.join(args.model_folder, f'checkpoint_best.pth')
                    save_model(model, file_name)
                    
        file_name = os.path.join(args.model_folder, f'checkpoint_final.pth')
        save_model(model, file_name)

    def test(self, model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1 

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output = model(data, _eval=True)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, 
                                                       pred if use_pseudo_label else label, 
                                                       'mean', 
                                                       False)

                    adv_output = model(adv_data, _eval=True)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num , total_adv_acc / num

def main(args):

    save_folder = '%s_%s' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

    attack = FastGradientSignUntargeted(model, 
                                        args.epsilon, 
                                        args.alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=args.k, 
                                        _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
                tv.transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
            ])
        tr_dataset_da = tv.datasets.CIFAR10(args.data_root, 
                                       train=True, 
                                       transform=transform_train, 
                                       download=True)
        
        tr_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=True, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)
        
        
        tr_dataset_split, val_dataset_split = __balance_val_split(tr_dataset_da, tr_dataset)
        
        tr_loader = DataLoader(tr_dataset_split, batch_size=args.batch_size, shuffle=True, num_workers=4)

        te_loader = DataLoader(val_dataset_split, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        te_dataset = tv.datasets.CIFAR10(args.data_root, 
                                       train=False, 
                                       transform=tv.transforms.ToTensor(), 
                                       download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)

        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False)

        print(f"std acc: {std_acc * 100:.3f}%, adv_acc: {adv_acc * 100:.3f}%")

    else:
        raise NotImplementedError




if __name__ == '__main__':
    args = parser()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
