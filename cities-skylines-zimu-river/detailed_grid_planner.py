"""
Cities: Skylines - 子母河地图详细网格规划
包含道路、建筑类型、服务设施的精细化规划
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class CellType(Enum):
    """网格单元类型"""
    # 地形
    WATER = 'water'
    LAND = 'land'
    FOREST = 'forest'

    # 道路
    HIGHWAY = 'highway'
    MAIN_ROAD = 'main_road'
    SECONDARY_ROAD = 'secondary_road'
    LOCAL_ROAD = 'local_road'

    # 住宅区
    RES_LOW = 'res_low'          # 低密度住宅
    RES_HIGH = 'res_high'        # 高密度住宅

    # 商业区
    COM_LOW = 'com_low'          # 低密度商业
    COM_HIGH = 'com_high'        # 高密度商业

    # 工业区
    IND_GENERIC = 'ind_generic'  # 普通工业
    IND_SPECIAL = 'ind_special'  # 特色工业

    # 办公区
    OFFICE = 'office'

    # 服务设施
    POWER_PLANT = 'power_plant'  # 电厂
    WATER_PUMP = 'water_pump'    # 水泵
    SEWAGE = 'sewage'            # 污水处理
    POLICE = 'police'            # 警察局
    FIRE = 'fire'                # 消防局
    HOSPITAL = 'hospital'        # 医院
    SCHOOL = 'school'            # 学校
    PARK = 'park'                # 公园

    # 交通设施
    BRIDGE = 'bridge'            # 桥梁
    BUS_STOP = 'bus_stop'        # 公交站

    EMPTY = 'empty'              # 空地


@dataclass
class GridCell:
    """网格单元"""
    x: int
    y: int
    cell_type: CellType
    priority: int = 1  # 1-5
    phase: int = 0     # 0-未规划, 1-初期, 2-中期, 3-后期
    notes: str = ""

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'type': self.cell_type.value,
            'priority': self.priority,
            'phase': self.phase,
            'notes': self.notes
        }


class DetailedGridPlanner:
    """详细网格规划器"""

    # 配色方案
    COLORS = {
        # 地形
        CellType.WATER: '#4FC3F7',
        CellType.LAND: '#E8F5E9',
        CellType.FOREST: '#66BB6A',

        # 道路
        CellType.HIGHWAY: '#263238',
        CellType.MAIN_ROAD: '#455A64',
        CellType.SECONDARY_ROAD: '#78909C',
        CellType.LOCAL_ROAD: '#B0BEC5',

        # 住宅
        CellType.RES_LOW: '#A5D6A7',
        CellType.RES_HIGH: '#388E3C',

        # 商业
        CellType.COM_LOW: '#90CAF9',
        CellType.COM_HIGH: '#1976D2',

        # 工业
        CellType.IND_GENERIC: '#FFE082',
        CellType.IND_SPECIAL: '#F57C00',

        # 办公
        CellType.OFFICE: '#81D4FA',

        # 服务设施
        CellType.POWER_PLANT: '#FF6F00',
        CellType.WATER_PUMP: '#0288D1',
        CellType.SEWAGE: '#795548',
        CellType.POLICE: '#1565C0',
        CellType.FIRE: '#D32F2F',
        CellType.HOSPITAL: '#E91E63',
        CellType.SCHOOL: '#9C27B0',
        CellType.PARK: '#7CB342',

        # 交通
        CellType.BRIDGE: '#FF5722',
        CellType.BUS_STOP: '#FFA726',

        CellType.EMPTY: '#FAFAFA',
    }

    # 图标映射
    ICONS = {
        CellType.POWER_PLANT: '⚡',
        CellType.WATER_PUMP: '💧',
        CellType.SEWAGE: '🚰',
        CellType.POLICE: '👮',
        CellType.FIRE: '🚒',
        CellType.HOSPITAL: '🏥',
        CellType.SCHOOL: '🏫',
        CellType.PARK: '🌳',
        CellType.BRIDGE: '🌉',
        CellType.BUS_STOP: '🚌',
    }

    def __init__(self, width: int = 50, height: int = 50):
        """
        初始化详细规划器

        Args:
            width: 宽度（格）
            height: 高度（格）
        """
        self.width = width
        self.height = height
        self.grid: List[List[GridCell]] = []
        self.facilities: List[Dict] = []  # 重要设施列表
        self._initialize_grid()

    def _initialize_grid(self):
        """初始化网格"""
        for y in range(self.height):
            row = []
            for x in range(self.width):
                cell = GridCell(x, y, CellType.EMPTY)
                row.append(cell)
            self.grid.append(row)

    def set_cell(self, x: int, y: int, cell_type: CellType,
                 priority: int = 1, phase: int = 0, notes: str = ""):
        """设置单个单元格"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].cell_type = cell_type
            self.grid[y][x].priority = priority
            self.grid[y][x].phase = phase
            self.grid[y][x].notes = notes

    def set_area(self, x1: int, y1: int, x2: int, y2: int,
                 cell_type: CellType, priority: int = 1, phase: int = 0, notes: str = ""):
        """设置矩形区域"""
        for y in range(y1, min(y2 + 1, self.height)):
            for x in range(x1, min(x2 + 1, self.width)):
                self.set_cell(x, y, cell_type, priority, phase, notes)

    def draw_road(self, points: List[Tuple[int, int]], road_type: CellType, phase: int = 1):
        """绘制道路"""
        for x, y in points:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.set_cell(x, y, road_type, priority=5, phase=phase)

    def draw_horizontal_road(self, y: int, x_start: int, x_end: int,
                            road_type: CellType, phase: int = 1):
        """绘制横向道路"""
        for x in range(x_start, x_end + 1):
            self.set_cell(x, y, road_type, priority=5, phase=phase)

    def draw_vertical_road(self, x: int, y_start: int, y_end: int,
                          road_type: CellType, phase: int = 1):
        """绘制纵向道路"""
        for y in range(y_start, y_end + 1):
            self.set_cell(x, y, road_type, priority=5, phase=phase)

    def add_facility(self, x: int, y: int, facility_type: CellType,
                    name: str, phase: int = 1):
        """添加重要设施"""
        self.set_cell(x, y, facility_type, priority=5, phase=phase, notes=name)
        self.facilities.append({
            'x': x, 'y': y,
            'type': facility_type.value,
            'name': name,
            'phase': phase
        })

    def draw_river(self, points: List[Tuple[int, int]], width: int = 3):
        """绘制河流"""
        for x, y in points:
            for dy in range(-width // 2, width // 2 + 1):
                for dx in range(-width // 2, width // 2 + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.set_cell(nx, ny, CellType.WATER)

    def visualize(self, show_grid: bool = True, show_icons: bool = True):
        """生成详细可视化"""
        fig, ax = plt.subplots(figsize=(20, 20))

        # 绘制网格背景
        grid_array = np.zeros((self.height, self.width, 3))

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                color = self.COLORS.get(cell.cell_type, '#FFFFFF')
                # 转换颜色
                rgb = self._hex_to_rgb(color)
                grid_array[y, x] = rgb

        ax.imshow(grid_array, interpolation='nearest', origin='upper')

        # 绘制网格线
        if show_grid:
            for x in range(self.width + 1):
                ax.axvline(x - 0.5, color='white', linewidth=0.3, alpha=0.5)
            for y in range(self.height + 1):
                ax.axhline(y - 0.5, color='white', linewidth=0.3, alpha=0.5)

        # 绘制设施图标
        if show_icons:
            for facility in self.facilities:
                x, y = facility['x'], facility['y']
                ftype = CellType(facility['type'])
                icon = self.ICONS.get(ftype, '●')

                # 绘制图标背景
                circle = plt.Circle((x, y), 0.4, color='white', alpha=0.9, zorder=10)
                ax.add_patch(circle)

                # 绘制图标
                ax.text(x, y, icon, ha='center', va='center',
                       fontsize=12, zorder=11)

                # 绘制标签
                ax.text(x, y - 1, facility['name'], ha='center', va='top',
                       fontsize=7, color='black', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', alpha=0.8, edgecolor='none'),
                       zorder=11)

        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xlabel('X (East →)', fontsize=12)
        ax.set_ylabel('Y (South ↓)', fontsize=12)
        ax.set_title('Two Rivers Map - Detailed Grid Plan with Buildings & Roads',
                    fontsize=16, fontweight='bold', pad=20)

        # 添加图例
        self._add_legend(ax)

        plt.tight_layout()
        return fig

    def _add_legend(self, ax):
        """添加图例"""
        legend_items = [
            # 地形
            ('Water', self.COLORS[CellType.WATER]),
            ('Forest', self.COLORS[CellType.FOREST]),
            # 道路
            ('Highway', self.COLORS[CellType.HIGHWAY]),
            ('Main Road', self.COLORS[CellType.MAIN_ROAD]),
            ('Local Road', self.COLORS[CellType.LOCAL_ROAD]),
            # 区域
            ('Residential (L)', self.COLORS[CellType.RES_LOW]),
            ('Residential (H)', self.COLORS[CellType.RES_HIGH]),
            ('Commercial', self.COLORS[CellType.COM_LOW]),
            ('Industrial', self.COLORS[CellType.IND_GENERIC]),
            ('Office', self.COLORS[CellType.OFFICE]),
            ('Park', self.COLORS[CellType.PARK]),
        ]

        legend_elements = [patches.Patch(facecolor=color, label=label)
                          for label, color in legend_items]

        ax.legend(handles=legend_elements, loc='upper right',
                 fontsize=9, ncol=2, framealpha=0.9)

    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """转换十六进制颜色到RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def save_json(self, filename: str):
        """保存到JSON"""
        data = {
            'width': self.width,
            'height': self.height,
            'grid': [[cell.to_dict() for cell in row] for row in self.grid],
            'facilities': self.facilities
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_facilities_list(self):
        """打印设施清单"""
        print("\n" + "="*80)
        print("重要设施清单 (Key Facilities)")
        print("="*80)

        phases = {1: [], 2: [], 3: []}
        for f in self.facilities:
            phases[f['phase']].append(f)

        for phase in [1, 2, 3]:
            if phases[phase]:
                print(f"\n【阶段 {phase}】")
                for f in phases[phase]:
                    print(f"  {f['name']:20s} @ ({f['x']:2d}, {f['y']:2d})")


def create_detailed_two_rivers_plan():
    """创建子母河地图的详细规划"""

    planner = DetailedGridPlanner(width=50, height=50)

    # ========== 地形 ==========

    # 绘制主河道
    for x in range(35, 42):
        for y in range(50):
            planner.set_cell(x, y, CellType.WATER)

    # 河流弯曲部分
    for y in range(18, 30):
        planner.set_cell(34, y, CellType.WATER)
        planner.set_cell(33, y, CellType.WATER)

    # 森林区域（左上角）
    planner.set_area(0, 0, 10, 10, CellType.FOREST)

    # ========== 阶段1: 初期道路网络 ==========

    # 主干道1（横向）- 连接高速出口到城市中心
    planner.draw_horizontal_road(15, 0, 32, CellType.MAIN_ROAD, phase=1)

    # 主干道2（横向）- 南部主干道
    planner.draw_horizontal_road(30, 0, 32, CellType.MAIN_ROAD, phase=1)

    # 主干道3（纵向）- 西侧南北主干道
    planner.draw_vertical_road(8, 11, 40, CellType.MAIN_ROAD, phase=1)

    # 次干道（网格状）
    planner.draw_vertical_road(12, 12, 25, CellType.SECONDARY_ROAD, phase=1)
    planner.draw_vertical_road(16, 12, 25, CellType.SECONDARY_ROAD, phase=1)
    planner.draw_horizontal_road(20, 0, 32, CellType.SECONDARY_ROAD, phase=1)

    # 高速公路连接
    for i in range(15):
        planner.set_cell(42 + i // 4, i, CellType.HIGHWAY, phase=1)

    # 桥梁
    planner.set_cell(33, 15, CellType.BRIDGE, priority=5, phase=1, notes="主桥")
    planner.set_cell(33, 30, CellType.BRIDGE, priority=5, phase=2, notes="南桥")

    # ========== 阶段1: 初期建筑 ==========

    # 起始住宅区（低密度）
    planner.set_area(9, 16, 11, 19, CellType.RES_LOW, priority=5, phase=1,
                    notes="起始住宅")
    planner.set_area(13, 16, 15, 19, CellType.RES_LOW, priority=5, phase=1)
    planner.set_area(9, 21, 11, 24, CellType.RES_LOW, priority=5, phase=1)
    planner.set_area(13, 21, 15, 24, CellType.RES_LOW, priority=5, phase=1)

    # 初期商业区
    planner.set_area(17, 16, 19, 19, CellType.COM_LOW, priority=5, phase=1,
                    notes="商业街")
    planner.set_area(17, 21, 19, 23, CellType.COM_LOW, priority=5, phase=1)

    # 工业区
    planner.set_area(3, 32, 7, 37, CellType.IND_GENERIC, priority=4, phase=1,
                    notes="工业区")
    planner.set_area(9, 32, 11, 37, CellType.IND_GENERIC, priority=4, phase=1)

    # ========== 阶段1: 基础设施 ==========

    # 风力发电厂
    planner.add_facility(2, 12, CellType.POWER_PLANT, "Wind Farm", phase=1)

    # 水泵站（河边）
    planner.add_facility(30, 10, CellType.WATER_PUMP, "Water Pump", phase=1)

    # 污水处理（下游）
    planner.add_facility(30, 35, CellType.SEWAGE, "Sewage Plant", phase=1)

    # 警察局
    planner.add_facility(14, 14, CellType.POLICE, "Police", phase=1)

    # 消防局
    planner.add_facility(10, 14, CellType.FIRE, "Fire Station", phase=1)

    # 小学
    planner.add_facility(20, 18, CellType.SCHOOL, "Elementary", phase=1)

    # 诊所
    planner.add_facility(18, 14, CellType.HOSPITAL, "Clinic", phase=1)

    # 公交站
    planner.add_facility(17, 15, CellType.BUS_STOP, "Bus Stop", phase=1)

    # ========== 阶段2: 中期扩张 ==========

    # 北部住宅区（高密度）
    planner.set_area(9, 5, 12, 9, CellType.RES_HIGH, priority=4, phase=2,
                    notes="高密度住宅")
    planner.set_area(14, 5, 17, 9, CellType.RES_HIGH, priority=4, phase=2)

    # CBD办公区
    planner.set_area(21, 12, 25, 14, CellType.OFFICE, priority=4, phase=2,
                    notes="CBD")
    planner.set_area(21, 16, 25, 18, CellType.OFFICE, priority=4, phase=2)

    # 高密度商业
    planner.set_area(27, 12, 30, 14, CellType.COM_HIGH, priority=4, phase=2,
                    notes="商业中心")

    # 中央公园
    planner.set_area(22, 6, 26, 9, CellType.PARK, priority=3, phase=2,
                    notes="中央公园")

    # 中学
    planner.add_facility(25, 10, CellType.SCHOOL, "High School", phase=2)

    # 大医院
    planner.add_facility(28, 10, CellType.HOSPITAL, "Hospital", phase=2)

    # ========== 阶段3: 后期发展 ==========

    # 南部住宅区
    planner.set_area(13, 32, 16, 36, CellType.RES_LOW, priority=2, phase=3)
    planner.set_area(18, 32, 21, 36, CellType.RES_LOW, priority=2, phase=3)

    # 南部商业
    planner.set_area(23, 32, 26, 35, CellType.COM_LOW, priority=2, phase=3)

    # 特色工业
    planner.set_area(3, 39, 7, 43, CellType.IND_SPECIAL, priority=2, phase=3,
                    notes="特色工业")

    # 大学
    planner.add_facility(28, 33, CellType.SCHOOL, "University", phase=3)

    return planner


if __name__ == "__main__":
    print("正在生成详细网格规划...")

    # 创建规划
    planner = create_detailed_two_rivers_plan()

    # 打印设施清单
    planner.print_facilities_list()

    # 保存JSON
    planner.save_json('detailed_grid_plan.json')
    print("\n✓ 详细规划已保存到 detailed_grid_plan.json")

    # 生成可视化
    fig = planner.visualize(show_grid=True, show_icons=True)
    plt.savefig('detailed_grid_plan.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 详细网格图已保存到 detailed_grid_plan.png")

    plt.close()

    print("\n完成！")
