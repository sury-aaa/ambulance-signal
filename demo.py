import random
import pygame
from signal_controller import SignalController, GREEN, YELLOW, RED

WIDTH, HEIGHT = 900, 700
LANE_X = {"Lane 1": 225, "Lane 2": 450, "Lane 3": 675}
STOP_Y = 430
CAR_W, CAR_H = 44, 80
GAP = 14
SPEED = 140             # pixels per second
DETECT_RANGE = 330      # how far before the stop line the "camera" can see
CONFIRM_FRAMES = 10     # ambulance must be seen this many frames in a row

LIGHT = {GREEN: (40, 200, 70), YELLOW: (250, 210, 40), RED: (220, 50, 50)}
CAR_COLOURS = [(70, 100, 180), (130, 130, 140), (60, 60, 70),
               (160, 70, 70), (90, 150, 110), (200, 160, 60)]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ambulance Signal Demo")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 20)
big = pygame.font.SysFont("arial", 28, bold=True)

lanes = list(LANE_X)
controller = SignalController(lanes, green_time=8, yellow_time=3,
                              all_red_time=1, priority_hold=3)
cars = {lane: [] for lane in lanes}
spawn_timer = {lane: random.uniform(0.2, 1.5) for lane in lanes}
amb_streak = 0


def must_stop(signal, front_y):
    if signal == RED:
        return True
    if signal == YELLOW:
        return STOP_Y - front_y > 30     # too far back to clear the line
    return False


def update_cars(lane, dt, signal):
    queue = sorted(cars[lane], key=lambda c: -c["y"])    # front car first
    ahead = None
    for car in queue:
        new_y = car["y"] + SPEED * dt
        front = car["y"] + CAR_H
        if front <= STOP_Y + 0.01 and must_stop(signal, front):
            new_y = min(new_y, STOP_Y - CAR_H)
        if ahead is not None:
            new_y = min(new_y, ahead["y"] - CAR_H - GAP)
        car["y"] = max(car["y"], new_y)                  # never move backwards
        ahead = car
    cars[lane] = [c for c in cars[lane] if c["y"] < HEIGHT]


def spawn(lane, amb=False):
    min_y = min((c["y"] for c in cars[lane]), default=HEIGHT)
    if not amb and min_y < GAP:
        return                                           # no room at the top
    y = min(-CAR_H, min_y - CAR_H - GAP)
    cars[lane].append({"y": y, "amb": amb, "colour": random.choice(CAR_COLOURS)})


def detect_ambulance():
    """Stand-in for the YOLO camera: sees an ambulance approaching the stop line."""
    for lane in lanes:
        for c in cars[lane]:
            front = c["y"] + CAR_H
            if c["amb"] and STOP_Y - DETECT_RANGE <= front <= STOP_Y + 0.01:
                return lane
    return None


def draw_car(car, lane):
    x = LANE_X[lane] - CAR_W // 2
    y = int(car["y"])
    rect = pygame.Rect(x, y, CAR_W, CAR_H)
    if car["amb"]:
        pygame.draw.rect(screen, (245, 245, 245), rect, border_radius=6)
        cx, cy = x + CAR_W // 2, y + CAR_H // 2
        pygame.draw.rect(screen, (220, 40, 40), (cx - 3, cy - 12, 6, 24))
        pygame.draw.rect(screen, (220, 40, 40), (cx - 12, cy - 3, 24, 6))
        flash = (pygame.time.get_ticks() // 250) % 2
        pygame.draw.rect(screen, (255, 0, 0) if flash else (0, 90, 255),
                         (x + 8, y + CAR_H - 12, CAR_W - 16, 7))
    else:
        pygame.draw.rect(screen, car["colour"], rect, border_radius=6)
        pygame.draw.rect(screen, (30, 30, 40), (x + 6, y + CAR_H - 28, CAR_W - 12, 14))


zone = pygame.Surface((CAR_W + 40, DETECT_RANGE), pygame.SRCALPHA)
zone.fill((80, 160, 255, 40))

running = True
while running:
    dt = clock.tick(60) / 1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                spawn(lanes[event.key - pygame.K_1], amb=True)

    for lane in lanes:                                   # normal traffic
        spawn_timer[lane] -= dt
        if spawn_timer[lane] <= 0:
            spawn(lane)
            spawn_timer[lane] = random.uniform(1.2, 2.8)

    amb_lane = detect_ambulance()
    amb_streak = amb_streak + 1 if amb_lane else 0
    priority = amb_lane if amb_streak >= CONFIRM_FRAMES else None
    signals = controller.update(priority_lane=priority)
    for lane in lanes:
        update_cars(lane, dt, signals[lane])

    # ---- drawing ----
    screen.fill((40, 44, 52))
    for lane, lx in LANE_X.items():
        pygame.draw.rect(screen, (70, 72, 78), (lx - 60, 0, 120, HEIGHT))
        screen.blit(zone, (lx - zone.get_width() // 2, STOP_Y - DETECT_RANGE))
        pygame.draw.line(screen, (240, 240, 240), (lx - 60, STOP_Y), (lx + 60, STOP_Y), 4)
        pygame.draw.circle(screen, LIGHT[signals[lane]], (lx + 80, STOP_Y - 10), 16)
        pygame.draw.circle(screen, (20, 20, 20), (lx + 80, STOP_Y - 10), 16, 2)
        screen.blit(font.render(lane, True, (230, 230, 230)), (lx - 30, 10))
        for car in cars[lane]:
            draw_car(car, lane)

    pygame.draw.rect(screen, (25, 27, 32), (0, HEIGHT - 70, WIDTH, 70))
    screen.blit(font.render(f"Controller state: {controller.state}", True, (230, 230, 230)),
                (20, HEIGHT - 60))
    screen.blit(font.render("Press 1 / 2 / 3 to send an ambulance into that lane   |   ESC to quit",
                            True, (160, 160, 170)), (20, HEIGHT - 32))
    if priority:
        banner = big.render(f"AMBULANCE DETECTED - {priority} PRIORITY", True, (255, 70, 70))
        screen.blit(banner, (WIDTH // 2 - banner.get_width() // 2, 45))

    pygame.display.flip()

pygame.quit()