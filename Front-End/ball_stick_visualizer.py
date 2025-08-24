import numpy as np
import py3Dmol

class BallStickVisualizer:
    """
    球棍模型分子可视化模块 - 使用 py3Dmol 进行高质量3D渲染
    """
    def __init__(self):
        pass

    def visualize_cluster(self, base_positions, predicted_sites=None, 
                         predicted_energies=None, best_site_idx=None,
                         title="Li团簇位点预测", save_path=None,
                         **kwargs):
        """
        使用 py3Dmol 高质量3D可视化Li团簇和预测位点。
        采用球棍模型、三维坐标轴和边框。
        """
        # 1. 准备数据
        xyz_data = ""
        atom_count = len(base_positions) + (len(predicted_sites) if predicted_sites is not None else 0)
        xyz_data += f"{atom_count}\n"
        xyz_data += f"{title}\n"

        all_pos = []
        for pos in base_positions:
            xyz_data += f"Li {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}\n"
            all_pos.append(pos)

        if predicted_sites is not None:
            # 根据能量对位点进行排序,以确定其等级
            ranks = {}
            if predicted_energies is not None and len(predicted_energies) == len(predicted_sites):
                sorted_indices = np.argsort(predicted_energies)
                ranks = {index: rank for rank, index in enumerate(sorted_indices)}
            else:
                # 如果没有提供能量信息,则假定列表已按等级排序
                ranks = {i: i for i in range(len(predicted_sites))}

            # 为不同等级的位点定义不同的化学元素符号作为标识
            # Top 5 + 其他
            elem_map = ['N', 'P', 'O', 'F', 'C'] 
            default_elem = 'S'

            for i, site in enumerate(predicted_sites):
                rank = ranks.get(i)
                if rank is not None and rank < len(elem_map):
                    elem = elem_map[rank]
                else:
                    elem = default_elem
                xyz_data += f"{elem} {site[0]:.4f} {site[1]:.4f} {site[2]:.4f}\n"
                all_pos.append(site)

        all_pos_np = np.array(all_pos)

        # 2. 创建py3Dmol视图
        view = py3Dmol.view(width=1000, height=800)

        # 3. 添加模型并设置球棍样式
        view.addModel(xyz_data, 'xyz')
        
        # 设置球棍模型样式 - 减小球的大小以突出棍的结构
        view.setStyle({'stick': {'radius': 0.15, 'color': 'gray'}, 'sphere': {'scale': 0.15}})

        # 定义科研调色板 (对色弱友好)
        # Rank 1: gold, Rank 2: blue, Rank 3: orange, Rank 4: green, Rank 5: red, Others: grey
        color_map = {
            'N': 'gold',    
            'P': '#0077BB', 
            'O': '#EE7733', 
            'F': '#009988', 
            'C': '#CC3311', 
            'S': '#BBBBBB'
        }

        # 为Li原子设置特殊颜色和样式 - 减小球的大小
        view.setStyle({'elem': 'Li'}, {
            'sphere': {'color': '#CC80FF', 'scale': 0.25}, 
            'stick': {'radius': 0.15, 'color': '#CC80FF'}
        })
        
        # 为预测位点设置颜色和样式 - 减小球的大小
        for elem, color in color_map.items():
            view.setStyle({'elem': elem}, {
                'sphere': {'color': color, 'opacity': 0.9, 'scale': 0.2}, 
                'stick': {'radius': 0.15, 'color': color}
            })

        # 4. 添加化学键连接
        if len(all_pos) > 1:
            # 计算原子间距离并添加键
            threshold = 3.5  # 键长阈值 (Å)
            for i in range(len(all_pos)):
                for j in range(i + 1, len(all_pos)):
                    dist = np.linalg.norm(np.array(all_pos[i]) - np.array(all_pos[j]))
                    if dist <= threshold:
                        view.addCylinder({
                            'start': {'x': all_pos[i][0], 'y': all_pos[i][1], 'z': all_pos[i][2]},
                            'end': {'x': all_pos[j][0], 'y': all_pos[j][1], 'z': all_pos[j][2]},
                            'radius': 0.08,
                            'color': 'gray',
                            'opacity': 0.7
                        })

        # 5. 添加三维边框和坐标轴
        if all_pos_np.size > 0:
            min_c = np.min(all_pos_np, axis=0)
            max_c = np.max(all_pos_np, axis=0)
            center = (min_c + max_c) / 2.0
            dims = (max_c - min_c) + np.array([2.0, 2.0, 2.0])
            
            # 添加线框盒子
            view.addBox({
                'center': {'x': center[0], 'y': center[1], 'z': center[2]},
                'dimensions': {'w': dims[0], 'h': dims[1], 'd': dims[2]},
                'color': 'black',
                'wireframe': True, # 使用线框模式
                'opacity': 0.6
            })

            # 添加XYZ坐标轴
            axis_len = np.max(dims) * 0.6
            origin = min_c - np.array([1.0, 1.0, 1.0]) # 将坐标轴放在角落
            # X轴 (红色)
            view.addArrow({'start': dict(x=origin[0], y=origin[1], z=origin[2]), 'end': dict(x=origin[0] + axis_len, y=origin[1], z=origin[2]), 'color': 'red', 'radius': 0.1})
            view.addLabel("X", {'position': {'x': origin[0] + axis_len, 'y': origin[1], 'z': origin[2]}, 'fontColor': 'red', 'fontSize': 14})
            # Y轴 (绿色)
            view.addArrow({'start': dict(x=origin[0], y=origin[1], z=origin[2]), 'end': dict(x=origin[0], y=origin[1] + axis_len, z=origin[2]), 'color': 'green', 'radius': 0.1})
            view.addLabel("Y", {'position': {'x': origin[0], 'y': origin[1] + axis_len, 'z': origin[2]}, 'fontColor': 'green', 'fontSize': 14})
            # Z轴 (蓝色)
            view.addArrow({'start': dict(x=origin[0], y=origin[1], z=origin[2]), 'end': dict(x=origin[0], y=origin[1], z=origin[2] + axis_len), 'color': 'blue', 'radius': 0.1})
            view.addLabel("Z", {'position': {'x': origin[0], 'y': origin[1], 'z': origin[2] + axis_len}, 'fontColor': 'blue', 'fontSize': 14})

        # 6. 设置缩放和慢速旋转
        view.zoomTo()
        view.spin(True)  # 默认启用自动旋转
        html = view.write_html()
        
        # 7. 添加旋转控制脚本 - 使用全局函数方法
        rotation_script = """
        <script>
        // 全局旋转控制函数
        window.toggleMoleculeRotation = function(shouldRotate) {
            console.log('toggleMoleculeRotation called with:', shouldRotate);
            
            // 尝试找到viewer实例
            var viewers = [];
            
            // 方法1: 通过$3Dmol.viewers
            if (window.$3Dmol && window.$3Dmol.viewers) {
                viewers = Object.values(window.$3Dmol.viewers);
            }
            
            // 方法2: 查找以viewer_开头的全局变量
            for (var prop in window) {
                if (prop.startsWith('viewer_') && window[prop] && typeof window[prop].spin === 'function') {
                    viewers.push(window[prop]);
                }
            }
            
            console.log('找到viewer数量:', viewers.length);
            
            viewers.forEach((viewer, index) => {
                try {
                    console.log(`处理viewer ${index}, 设置旋转:`, shouldRotate);
                    viewer.spin(shouldRotate);
                    viewer.render();
                    console.log(`viewer ${index} 旋转状态已更新`);
                } catch(e) {
                    console.error(`处理viewer ${index} 失败:`, e);
                }
            });
            
            return viewers.length > 0;
        };
        
        // 旋转控制功能 (保留postMessage支持)
        window.addEventListener('message', function(event) {
            console.log('收到postMessage:', event.data);
            if (event.data.action === 'stopSpin') {
                window.toggleMoleculeRotation(false);
            } else if (event.data.action === 'startSpin') {
                window.toggleMoleculeRotation(true);
            }
        });
        
        // 当页面加载完成后通知父窗口并设置全局引用
        function setupRotationControl() {
            console.log('设置旋转控制...');
            
            // 将函数暴露到父窗口
            if (window.parent !== window) {
                try {
                    window.parent.molViewerControl = window.toggleMoleculeRotation;
                    console.log('已设置父窗口旋转控制函数');
                } catch(e) {
                    console.log('无法设置父窗口函数:', e.message);
                }
                
                window.parent.postMessage({action: 'viewerReady'}, '*');
                console.log('已通知父窗口viewer就绪');
            }
            
            // 调试信息
            if (window.$3Dmol && window.$3Dmol.viewers) {
                console.log('可用的3Dmol viewers:', Object.keys(window.$3Dmol.viewers));
            }
            
            for (var prop in window) {
                if (prop.startsWith('viewer_')) {
                    console.log('找到全局viewer变量:', prop);
                }
            }
        }
        
        // 等待页面完全加载
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                setTimeout(setupRotationControl, 2000);
            });
        } else {
            setTimeout(setupRotationControl, 2000);
        }
        </script>
        """
        
        # 在</body>之前插入脚本，如果没有</body>标签则追加到末尾
        if '</body>' in html:
            html = html.replace('</body>', rotation_script + '</body>')
        else:
            html = html + rotation_script

        # 7. 保存到文件
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html)
            print(f"交互式py3Dmol球棍模型图已保存到: {save_path}")

        return html