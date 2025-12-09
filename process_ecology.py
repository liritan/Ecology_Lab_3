# process_ecology.py 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import logging

from functions import pend, calculate_total_loss, fx_linear  
from radar_diagram import RadarDiagram

data_sol = []
logger = logging.getLogger(__name__)

def fill_diagrams(data, initial_equations, restrictions):
    radar = RadarDiagram()
    
    clipped_initial = np.clip(initial_equations, 0, 1.0)
    clipped_data = np.clip(data, 0, 1.0)
    clipped_restrictions = np.clip(restrictions, 0, 1.0)

    conc_indices = [
        0,                    # C = 0
        int(len(data) / 4),   # C = 0.25
        int(len(data) / 2),   # C = 0.5
        int(3 * len(data) / 4), # C = 0.75
        -1                    # C = 1.0
    ]
    
    titles = [
        "Характеристики при C = 0 (начальная концентрация)",
        "Характеристики при C = 0.25",
        "Характеристики при C = 0.5", 
        "Характеристики при C = 0.75",
        "Характеристики при C = 1.0 (максимальная концентрация)"
    ]
    
    filenames = [
        './static/images/diagram_eco.png',
        './static/images/diagram_eco2.png',
        './static/images/diagram_eco3.png',
        './static/images/diagram_eco4.png',
        './static/images/diagram_eco5.png'
    ]

    for i, (idx, title, fname) in enumerate(zip(conc_indices, titles, filenames)):
        current_vals = clipped_data[idx]
        
        if i == 0:
            radar.draw(
                filename=fname,
                initial_data=clipped_initial,
                current_data=current_vals,
                label="",
                title=title,
                restrictions=clipped_restrictions,
                show_both_lines=False
            )
        else:
            radar.draw(
                filename=fname,
                initial_data=clipped_initial,
                current_data=current_vals,
                label="",
                title=title,
                restrictions=clipped_restrictions,
                show_both_lines=True
            )

def create_graphic(C, data):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    labels = [
        "Cf₁ - Потери от заболеваемости населения",
        "Cf₂ - Потери сельского хозяйства", 
        "Cf₃ - Потери от изменения природной среды",
        "Cf₄ - Потери от ухудшения качества жизни",
        "Cf₅ - Потери предприятия"
    ]
    
    line_labels = ["$Cf_{1}$", "$Cf_{2}$", "$Cf_{3}$", "$Cf_{4}$", "$Cf_{5}$"]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    num_lines = 5
    label_positions_x = np.linspace(0.1, 0.4, num_lines)
    
    for i in range(5):
        y_data = np.clip(data[:, i], 0, 1.0)
        ax.plot(C, y_data, color=colors[i], linewidth=2.5, label=labels[i])
        x_pos = label_positions_x[i]
 
        closest_idx = np.argmin(np.abs(C - x_pos))
        if closest_idx < len(y_data):
            y_pos = y_data[closest_idx]
  
            if closest_idx > 0 and closest_idx < len(y_data) - 1:
                dy = y_data[closest_idx + 1] - y_data[closest_idx - 1]
                dx = C[closest_idx + 1] - C[closest_idx - 1]
                angle = np.degrees(np.arctan2(dy, dx)) if dx != 0 else 0
            else:
                angle = 0
     
            if angle > 45:
                angle = angle - 180
            elif angle < -45:
                angle = angle + 180
   
            if closest_idx > 0 and closest_idx < len(y_data) - 1:

                tx = C[closest_idx + 1] - C[closest_idx - 1]
                ty = y_data[closest_idx + 1] - y_data[closest_idx - 1]
                length = np.sqrt(tx*tx + ty*ty)
                if length > 0:
                    nx = -ty / length
                    ny = tx / length
                    offset_x = 0.00 * nx
                    offset_y = 0.00 * ny
                else:
                    offset_x = 0.00
                    offset_y = 0.00
            else:
                offset_x = 0.00
                offset_y = 0.00
            
            if i % 2 == 0:
                offset_x *= 1.2
                offset_y *= 1.2
            else:
                offset_x *= 0.8
                offset_y *= 0.8
            
            ax.text(x_pos + offset_x, y_pos + offset_y, f'{line_labels[i]}', 
                    color='black', fontsize=18, fontweight='bold',
                    va='center', ha='center',
                    rotation=angle,
                    bbox=None)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.0])
    # Увеличиваем и делаем подписи черными
    ax.set_xlabel("C, концентрация загрязняющих веществ", 
                 fontsize=18, fontweight='bold', color='black')
    ax.set_ylabel("Значения характеристик", 
                 fontsize=18, fontweight='bold', color='black')
    ax.set_title("График характеристик от концентрации загрязняющих веществ", 
                fontsize=20, fontweight='bold', pad=20, color='black')

    # Настраиваем легенду (оставляем как было)
    legend = ax.legend(loc='upper left', fontsize=14, framealpha=0.9, 
                      edgecolor='black', fancybox=True)
    plt.setp(legend.get_texts(), color='black')
    
    # Настраиваем сетку и оси
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=16, colors='black')
    
    # Настраиваем лимитную линию
    ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Предел')

    # Убираем верхнюю и правую границы
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    # Делаем подписи значений на осях черными
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('black')
        label.set_fontsize(14)
    
    plt.tight_layout(pad=3.0)
    fig.savefig('./static/images/figure_eco.png', bbox_inches='tight', dpi=150)
    plt.close(fig)


def cast_to_float(initial_equations, faks, equations, restrictions):
    for i in range(len(initial_equations)):
        initial_equations[i] = float(initial_equations[i])

    for i in range(len(faks)):
        for j in range(len(faks[i])):
            faks[i][j] = float(faks[i][j])

    for i in range(len(equations)):
        for j in range(len(equations[i])):
            equations[i][j] = float(equations[i][j])

    for i in range(len(restrictions)):
        restrictions[i] = float(restrictions[i])

    return initial_equations, faks, equations, restrictions


def process(initial_equations, faks, equations, restrictions, time_value=0.0):
    global data_sol

    initial_equations, faks, equations, restrictions = cast_to_float(
        initial_equations, faks, equations, restrictions
    )
    time_value = float(time_value)

    logger.info(f"Параметры внутренних функций получены с интерфейса:")
    for i, eq_params in enumerate(equations):
        if eq_params:  
            logger.info(f"  f{i+1}: {eq_params}")

    C = np.linspace(0, 1, 100)

    xm = [1.0, 1.0, 1.0, 1.0, 1.0] 

    data_sol = odeint(pend, initial_equations, C, args=(faks, equations, xm, time_value))
    
  
    data_sol = np.clip(data_sol, 0.0, 1.0) 
    create_graphic(C, data_sol)
    create_disturbances_graphic(C, faks, time_value)  
    fill_diagrams(data_sol, initial_equations, restrictions)
    
    logger.info(f"Расчет завершен. Концентрация: {len(C)} точек, время t={time_value}.")
    logger.info(f"Начальные значения: {initial_equations}")
    logger.info(f"Конечные значения при C=1: {data_sol[-1]}")
    
    logger.info("Значения возмущений x1-x6 в момент времени t=" + str(time_value) + ":")
    for i in range(min(6, len(faks))):
        value = fx_linear(time_value, faks[i])
        logger.info(f"  x{i+1}(t) = {value:.4f}")

u_list = [
    "Cf₁ - Потери, связанные с ростом заболеваемости населения",
    "Cf₂ - Потери сельского хозяйства от воздействия атмосферных поллютантов",
    "Cf₃ - Потери от изменения природной среды",
    "Cf₄ - Потери из-за ухудшения качества жизни населения",
    "Cf₅ - Потери предприятия, возникающие при регулировании атмосферных выбросов и оплате штрафов"
]

def create_disturbances_graphic(C, faks, time_value=0.0):
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    ax1, ax2, ax3 = axes
 
    colors_time = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    colors_conc_1 = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors_conc_2 = ['#ff1493', '#00ced1', '#ff7f0e', '#2ca02c']
    
    # ВОЗМУЩЕНИЯ, ЗАВИСЯЩИЕ ОТ ВРЕМЕНИ (x₁-x₆)
    x_positions_time = np.linspace(0.1, 0.9, 6)  # Равномерно распределяем по всей ширине
    
    # Сортируем значения по величине, чтобы избежать наложения
    values = []
    for i in range(min(6, len(faks))):
        if len(faks[i]) >= 2:
            value = max(0.0, min(1.0, fx_linear(time_value, faks[i])))
            values.append((i, value))
    
    # Сортируем по значению для лучшего распределения
    values.sort(key=lambda x: x[1])
    
    for idx, (i, value) in enumerate(values):
        eq_str = " = a·t + b"
        
        ax1.axhline(y=value, color=colors_time[i], linewidth=2.5, 
                   label=f"χ$_{{{i+1}}}$(t){eq_str}")

        # Позиционируем метки с небольшим смещением по вертикали, чтобы избежать наложения
        y_offset = 0.03 if idx % 2 == 0 else -0.03
        y_pos = max(0.05, min(0.95, value + y_offset))  # Удерживаем в пределах графика
        
        ax1.text(x_positions_time[idx], y_pos, f'$χ_{{{i+1}}}$', 
                color='black', fontsize=15, fontweight='bold',
                va='center', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                        edgecolor='none', alpha=0.9))
    
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.0])
    ax1.set_ylabel("Значение возмущения", fontsize=12, fontweight='bold')
    ax1.set_title(f"Возмущения, зависящие от времени (t = {time_value:.2f})", 
                 fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    # ВОЗМУЩЕНИЯ, ЗАВИСЯЩИЕ ОТ КОНЦЕНТРАЦИИ (x₇-x₁₀) 
    curves_1 = []
    for i in range(6, min(10, len(faks))):
        if i < len(faks) and len(faks[i]) >= 2:
            curve = []
            for c_val in C:
                value = fx_linear(c_val, faks[i])
                value = max(0.0, min(1.0, value))
                curve.append(value)
            curves_1.append((i, curve))
    
    num_curves_1 = len(curves_1)
    if num_curves_1 > 0:
        label_positions_x = np.linspace(0.1, 0.4, num_curves_1)

    offset_step = 0.05
    
    for idx, (i, curve) in enumerate(curves_1):
        curve = np.array(curve) + idx * offset_step
        curves_1[idx] = (i, curve)
        label_idx = i - 6
      
        eq_str = " = a·C + b"
        
        if label_idx < len(colors_conc_1):
            ax2.plot(C, curve, color=colors_conc_1[label_idx], linewidth=2.5, 
                    label=f"χ$_{{{i+1}}}$(C){eq_str}")
   
        x_pos = label_positions_x[idx]
        
        closest_idx = np.argmin(np.abs(C - x_pos))
        if closest_idx < len(curve):
            y_pos = curve[closest_idx]
            
            if closest_idx > 0 and closest_idx < len(curve) - 1:
                dy = curve[closest_idx + 1] - curve[closest_idx - 1]
                dx = C[closest_idx + 1] - C[closest_idx - 1]
                angle = np.degrees(np.arctan2(dy, dx)) if dx != 0 else 0
            else:
                angle = 0
         
            if angle > 45:
                angle = angle - 180
            elif angle < -45:
                angle = angle + 180
        
            if closest_idx > 0 and closest_idx < len(curve) - 1:
                tx = C[closest_idx + 1] - C[closest_idx - 1]
                ty = curve[closest_idx + 1] - curve[closest_idx - 1]
                length = np.sqrt(tx*tx + ty*ty)
                if length > 0:
                    nx = -ty / length
                    ny = tx / length
                    offset_x = 0.00 * nx
                    offset_y = 0.00 * ny
                else:
                    offset_x = 0.00
                    offset_y = 0.00
            else:
                offset_x = 0.02
                offset_y = 0.02
        
            if idx % 2 == 0:
                offset_x *= 1.2
                offset_y *= 1.2
            else:
                offset_x *= 0.8
                offset_y *= 0.8
            
            ax2.text(x_pos + offset_x, y_pos + offset_y, f'$χ_{{{i+1}}}$', 
                    color='black', fontsize=15, fontweight='bold',
                    va='center', ha='center',
                    rotation=angle,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor='none', alpha=0.9))
    
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel("Значение возмущения", fontsize=12, fontweight='bold')
    ax2.set_title("Возмущения, зависящие от концентрации (χ₇-χ₁₀)", 
                 fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylim(0, 1.5)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=1)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # ВОЗМУЩЕНИЯ, ЗАВИСЯЩИЕ ОТ КОНЦЕНТРАЦИИ (x₁₁-x₁₄) 
    curves_2 = []
    for i in range(10, min(14, len(faks))):
        if i < len(faks) and len(faks[i]) >= 2:
            curve = []
            for c_val in C:
                value = fx_linear(c_val, faks[i])
                value = max(0.0, min(1.0, value))
                curve.append(value)
            curves_2.append((i, curve))

    num_curves_2 = len(curves_2)
    if num_curves_2 > 0:
        label_positions_x_2 = np.linspace(0.1, 0.4, num_curves_2)
    
    offset_step = 0.04 
    
    for idx, (i, curve) in enumerate(curves_2):

        curve = np.array(curve) + idx * offset_step
        curves_1[idx] = (i, curve)
        label_idx = i - 10

        eq_str = " = a·C + b"
        
        if label_idx < len(colors_conc_2):
            ax3.plot(C, curve, color=colors_conc_2[label_idx], linewidth=2.5, 
                    label=f"χ$_{{{i+1}}}$(C){eq_str}")

        x_pos = label_positions_x_2[idx]

        closest_idx = np.argmin(np.abs(C - x_pos))
        if closest_idx < len(curve):
            y_pos = curve[closest_idx]
            

            if closest_idx > 0 and closest_idx < len(curve) - 1:
                dy = curve[closest_idx + 1] - curve[closest_idx - 1]
                dx = C[closest_idx + 1] - C[closest_idx - 1]
                angle = np.degrees(np.arctan2(dy, dx)) if dx != 0 else 0
            else:
                angle = 0

            if angle > 45:
                angle = angle - 180
            elif angle < -45:
                angle = angle + 180
            

            if closest_idx > 0 and closest_idx < len(curve) - 1:
                tx = C[closest_idx + 1] - C[closest_idx - 1]
                ty = curve[closest_idx + 1] - curve[closest_idx - 1]
                length = np.sqrt(tx*tx + ty*ty)
                if length > 0:
                    nx = -ty / length
                    ny = tx / length
                    offset_x = 0.00 * nx
                    offset_y = 0.00 * ny
                else:
                    offset_x = 0.00
                    offset_y = 0.00
            else:
                offset_x = 0.02
                offset_y = 0.02

            if idx % 2 == 0:
                offset_x *= 1.3
                offset_y *= 1.3
            else:
                offset_x *= 0.7
                offset_y *= 0.7
            
            ax3.text(x_pos + offset_x, y_pos + offset_y, f'$χ_{{{i+1}}}$', 
                    color='black', fontsize=15, fontweight='bold',
                    va='center', ha='center',
                    rotation=angle,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                            edgecolor='none', alpha=0.9))
    
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.0])
    ax3.set_xlabel("C, концентрация загрязняющих веществ", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Значение возмущения", fontsize=12, fontweight='bold')
    ax3.set_title("Возмущения, зависящие от концентрации (χ₁₁-χ₁₄)", 
                 fontsize=14, fontweight='bold', pad=10)
    ax3.set_ylim(0, 1.5)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=1)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(axis='both', which='major', labelsize=10)
    

    for ax in axes:
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.axhline(y=0.0, color='black', linestyle='-', alpha=0.1, linewidth=0.5)
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.1, linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig('./static/images/disturbances_eco.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
  
    logger.info(f"Создан график возмущений. t={time_value:.2f}")

    for i in range(min(6, len(faks))):
        if len(faks[i]) >= 2:
            value = fx_linear(time_value, faks[i])
            logger.info(f"  x{i+1}(t={time_value:.2f}) = {value:.4f}")

    all_values_in_range = True
    for i in range(len(faks)):
        if len(faks[i]) >= 2:
            test_val1 = fx_linear(0.0, faks[i])
            test_val2 = fx_linear(1.0, faks[i])
            
            if test_val1 < 0 or test_val1 > 1 or test_val2 < 0 or test_val2 > 1:
                logger.warning(f"Возмущение x{i+1} может выйти за пределы [0,1]: "
                             f"при 0={test_val1:.3f}, при 1={test_val2:.3f}")
                all_values_in_range = False
    
    if all_values_in_range:
        logger.info("Все возмущения находятся в диапазоне [0,1] ✓")