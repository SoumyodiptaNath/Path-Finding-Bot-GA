import pygame
import numpy as np
from icecream import ic
from data_file import x_screen, y_screen


class Gameplay():    
    def __init__(self, bot, num_obst_max = 10, obst_max_size = 20) -> None:
        self.bot_sim = bot
        self.num_obst_max = num_obst_max
        self.obst_max_size = obst_max_size
        self.bot_coord = lambda sign, index : self.bot_sim.pos + sign*self.bot_sim.radius*self.bot_sim.dir[index]
        # self.init_pygame()

    def init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Path Finding Bot")
        self.screen = pygame.display.set_mode((x_screen, y_screen))
        self.clock = pygame.time.Clock()
        
    def quit_pygame(self):
        pygame.quit()


    def get_obst(self, new=1):
        self.init_pygame()
        self.bot_sim.obst_bounds = np.array([[np.zeros(2), np.array([x_screen, 0.0])],
                                             [np.zeros(2), np.array([0.0, y_screen])],
                                             [np.array([x_screen, y_screen]), np.array([0.0, -y_screen])],
                                             [np.array([x_screen, y_screen]), np.array([-x_screen, 0.0])]])

        if new:
            count = 0
            running = True
            got_end_pt = True
            while running:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q]: running = False                

                self.draw_game()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        curr_start_pt = np.array(pygame.mouse.get_pos())
                        got_end_pt = False
                    
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and not got_end_pt:
                        curr_end_pt = np.array(pygame.mouse.get_pos())
                        line_dir = curr_end_pt-curr_start_pt
                        self.bot_sim.obst_bounds = np.append(self.bot_sim.obst_bounds, np.array([[curr_start_pt, line_dir]]), axis=0)
                        self.bot_sim.get_state(); got_end_pt = True; count+=1
                
                if count == self.num_obst_max: running = False

        self.quit_pygame()


    def text_box(self, text):
        font = pygame.font.Font('freesansbold.ttf', 20)
        disp_text = font.render(text, True, (255, 255, 255))
        textRect = disp_text.get_rect()
        textRect.center = (x_screen*0.75, y_screen*0.15)
        self.screen.blit(disp_text, textRect)


    def draw_game(self):
        self.screen.fill((25, 45, 40))

        pygame.draw.circle(self.screen, (0, 250, 0), self.bot_sim.start_pt, 10)
        pygame.draw.circle(self.screen, (250, 0, 250), self.bot_sim.end_pt, 10)

        for pt in self.bot_sim.int_pts:
            pygame.draw.line(self.screen, (250, 250, 250), self.bot_sim.pos, pt, 5)
        
        if len(self.bot_sim.pos_record) > 2:
            pygame.draw.lines(self.screen, (250, 250, 250), False, self.bot_sim.pos_record, 3)

        pygame.draw.circle(self.screen, (255, 190, 100), self.bot_sim.pos, self.bot_sim.radius)
        pygame.draw.circle(self.screen, (246, 92, 81), self.bot_sim.pos, self.bot_sim.radius, 3)
        pygame.draw.polygon(self.screen, (246, 92, 81), [self.bot_coord(-0.75, 0), self.bot_coord(0.75, 0), self.bot_coord(0.75, 1)])

        for line_info in self.bot_sim.obst_bounds:
            pygame.draw.line(self.screen, (229, 36, 63), line_info[0], line_info[0] + line_info[1], 5)
        
        self.text_box(f"V={self.bot_sim.v:.2f}, W={self.bot_sim.w:.2f}")
        pygame.display.update()
    

    def simulate(self, max_steps, dt_val=1):
        iter = 0
        dt = dt_val
        running = True
        self.init_pygame()
        pygame.time.delay(3000)

        while running:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]: running = False                

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            
            self.draw_game()

            if dt_val==1:
                dt = self.clock.tick(60)/1000
            
            self.bot_sim.control_bot(dt=dt)
            self.bot_sim.prev_state = np.copy(self.bot_sim.curr_state)
            self.bot_sim.get_state()

            if not self.bot_sim.is_safe or self.bot_sim.is_done or iter>max_steps:
                running = False
            iter += 1
            
        self.quit_pygame()
        return iter


    def draw_pos_record(self, gen):
        self.screen.fill((25, 45, 40))
        
        for line_info in self.bot_sim.obst_bounds:
            pygame.draw.line(self.screen, (229, 36, 63), line_info[0], line_info[0] + line_info[1], 5)
        pygame.draw.circle(self.screen, (0, 250, 0), self.bot_sim.start_pt, 10)
        pygame.draw.circle(self.screen, (250, 0, 250), self.bot_sim.end_pt, 10)
        
        iter = 0
        max_dist = 0
        curr_len = []
        best_index = 0
        running = True
        pop_size = len(gen)

        for i, path in enumerate(gen):
            curr_len.append(len(path))
            curr_dist = np.linalg.norm(self.bot_sim.start_pt - path[-1])
            if curr_dist > max_dist:
                max_dist = curr_dist
                best_index = i

        while running:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]: 
                running = False
                return running

            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    running = False
                    return running
            
            count = 0
            for i, path in enumerate(gen):
                if curr_len[i] > iter:
                    pygame.draw.circle(self.screen, (250, 250, 250), path[iter], 1, 1)
                else:
                    count += 1
            
            iter += 1
            pygame.display.update()
            if count == pop_size: running = False
        
        pygame.draw.lines(self.screen, (0, 250, 250), False, gen[best_index], 7)
        pygame.display.update()
        pygame.time.delay(2000)
        return True
