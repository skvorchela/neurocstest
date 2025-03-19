from demoinfocs import DemoParser # type: ignore
import pandas as pd
import os
import glob

YOUR_ACCOUNTS = ["76561199080599486", "skvorchella","76561199077798899", "sevtladder","sevt_eagle", "76561199077655702","76561199080378044", "76561199080621789"]  # Замените!

def parse_demo(demo_path: str) -> pd.DataFrame:
    df_entries = []
    try:
        with DemoFile(demo_path) as demo:
            demo.parse()
            for player in demo.entities.players:
                if player.steam_id not in YOUR_ACCOUNTS:
                    continue  # Пропускаем чужие аккаунты
                
                for tick in demo.ticks[::4]:
                    demo.seek_to_tick(tick)
                    if not player.is_alive:
                        continue

                    # Состояние игры (полный набор параметров)
                    game_state = {
                        "steamid": player.steam_id,
                        "tick": tick,
                        "map": demo.map_name,
                        "health": player.health,
                        "armor": player.armor,  # Добавлено
                        "x": player.x,
                        "y": player.y,
                        "z": player.z,
                        "weapon": player.active_weapon.class_name if player.active_weapon else "none",
                        "ammo": player.active_weapon.ammo if player.active_weapon else 0,  # Добавлено
                        "enemies_visible": sum(1 for p in demo.entities.players 
                                              if p.team != player.team and p.is_alive and p.is_visible),
                        "bomb_planted": demo.game_rules.bomb_planted,  # Добавлено
                    }

                    # Действия игрока (полный набор)
                    actions = {
                        "move_forward": (player.buttons & (1 << 0)) != 0,
                        "move_backward": (player.buttons & (1 << 1)) != 0,
                        "move_left": (player.buttons & (1 << 2)) != 0,
                        "move_right": (player.buttons & (1 << 3)) != 0,
                        "shoot": (player.buttons & (1 << 4)) != 0,
                        "jump": (player.buttons & (1 << 5)) != 0,  # Добавлено
                        "crouch": (player.buttons & (1 << 6)) != 0,  # Добавлено
                        "reload": (player.buttons & (1 << 7)) != 0,  # Добавлено
                        "aim_x": player.view_angle_x,
                        "aim_y": player.view_angle_y,
                        "zoom": player.fov,  # Добавлено (уровень зума)
                    }

                    df_entries.append({**game_state, **actions})
                    
        return pd.DataFrame(df_entries)
    
    except Exception as e:
        print(f"Ошибка в {os.path.basename(demo_path)}: {str(e)}")
        return pd.DataFrame()

# Остальной код (пути, сохранение и т.д.)