# train_ddp.py (已加入DDP同步逻辑，可彻底解决竞争问题)
import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from li_dataset import LiClusterDataset
from model import SchNetModel

def setup_ddp(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """清理分布式环境"""
    dist.destroy_process_group()

def main(rank, world_size, args):
    # 减少调试信息输出
    if rank == 0:
        print(f"在 Rank {rank} 上运行 DDP 训练，总进程数: {world_size}。")
    setup_ddp(rank, world_size)

    # ######################################################################
    # ## 这是解决问题的关键：确保只有 Rank 0 处理数据，其他进程等待 ##
    # ######################################################################
    if rank == 0:
        # 主进程负责检查数据和预处理。
        # 如果 li_dataset_processed 文件夹不存在，这行代码会触发 .process()
        print("正在初始化数据集，如果需要，将执行预处理...")
        LiClusterDataset(root=args.data_root, data_path=args.data_file)
        print("数据集检查/处理完成。")

    # 设置一个屏障，所有进程必须在这里等待，直到 rank 0 完成上面的任务
    dist.barrier()
    
    # 屏障被打破后，所有进程都可以安全地初始化数据集了，
    # 因为它们现在可以确定预处理已经完成，可以直接从磁盘加载。
    dataset = LiClusterDataset(root=args.data_root, data_path=args.data_file)
    if rank == 0:
        print("数据集加载成功。")
    # ######################################################################
    # ## 同步逻辑结束 ##
    # ######################################################################

    # --- 数据准备 ---
    # 使用标准的 70%/15%/15% 划分
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    if rank == 0:
        print(f"数据划分: 训练集={train_size} ({train_size/total_size*100:.1f}%), "
              f"验证集={val_size} ({val_size/total_size*100:.1f}%), "
              f"测试集={test_size} ({test_size/total_size*100:.1f}%)")
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    if rank == 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- 模型、损失函数和优化器 ---
    model = SchNetModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False) # find_unused_parameters 设为False可提速
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
    
    best_val_mae = float('inf')

    # --- 训练循环 ---
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        
        pbar = None
        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch in train_loader:
            batch = batch.to(rank)
            optimizer.zero_grad()
            pred = ddp_model(batch)
            loss = loss_fn(pred.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            
            if rank == 0:
                pbar.update(1)
        
        if rank == 0:
            pbar.close()

        # --- 验证与模型保存 (只在主进程执行) ---
        if rank == 0:
            ddp_model.eval()
            total_val_mae = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                    batch = batch.to(rank)
                    pred_normalized = ddp_model.module(batch)
                    pred_real = pred_normalized * dataset.std + dataset.mean
                    true_real = batch.y * dataset.std + dataset.mean
                    total_val_mae += torch.nn.functional.l1_loss(pred_real.squeeze(), true_real).item()
            
            avg_val_mae = total_val_mae / len(val_loader)
            print(f"Epoch {epoch+1}, Avg Val MAE (Real): {avg_val_mae:.6f} eV")

            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                torch.save(ddp_model.module.state_dict(), 'best_schnet_model.pt')
                print(f"*** 新的最佳模型已保存，验证 MAE: {best_val_mae:.6f} eV ***")

    # --- 训练结束后在测试集上评估最佳模型 ---
    if rank == 0:
        print("\n" + "="*60)
        print("训练完成，正在测试集上评估最佳模型...")
        
        # 加载最佳模型
        ddp_model.module.load_state_dict(torch.load('best_schnet_model.pt'))
        ddp_model.eval()
        
        total_test_mae = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = batch.to(rank)
                pred_normalized = ddp_model.module(batch)
                pred_real = pred_normalized * dataset.std + dataset.mean
                true_real = batch.y * dataset.std + dataset.mean
                total_test_mae += torch.nn.functional.l1_loss(pred_real.squeeze(), true_real).item()
        
        avg_test_mae = total_test_mae / len(test_loader)
        print(f"\n🎯 最终测试集 MAE: {avg_test_mae:.6f} eV")
        print(f"📊 最佳验证集 MAE: {best_val_mae:.6f} eV")
        print(f"📈 泛化性能 (测试-验证): {avg_test_mae - best_val_mae:+.6f} eV")
        print("="*60)

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed SchNet Training")
    parser.add_argument('--data_root', type=str, default='./li_dataset_processed', help='文件夹，用于存放处理好的数据')
    parser.add_argument('--data_file', type=str, default='/data/Hackthon/data/TheDataOfClusters_4_40 copy.data', help='原始数据文件的路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='每个GPU的批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size > 0:
        print(f"检测到 {world_size} 张 GPU，开始分布式训练。")
        torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        print("未检测到GPU，请检查环境配置。")