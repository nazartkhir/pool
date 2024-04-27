import cv2
import numpy as np
from sympy import *
from sympy.geometry import *


class Shot():
    def __init__(self, ball1, ball2, lusa, balls, field, path) -> None:
        self.balls = balls
        self.path = path
        self.field = field
        self.cue = ball1
        self.target = ball2
        self.lusa = lusa
        self.balls_dist = Segment2D(ball1.center, ball2.center)
        self.lusa_dist = Segment2D(lusa.midpoint, ball2.center)
        self.lusa_angle = self.lusa_ang()
        self.angle = float(Ray2D(ball1.center, ball2.center).angle_between(Ray2D(ball2.center, lusa.midpoint))) * 360 /(2*3.1415)
        self.score = 100000
    def lusa_ang(self):
        line = self.lusa.perpendicular_line(self.lusa.midpoint)
        return float(line.smallest_angle_between(self.lusa_dist)) * 360/(2*3.1415)
    def paint(self):
        img = cv2.imread('intermtop_view.png')
        cv2.line(img, (int(self.cue.center.x), int(self.cue.center.y)), (int(self.target.center.x), int(self.target.center.y)),(0,0,255), 3)
        cv2.line(img, (int(self.lusa.midpoint.x), int(self.lusa.midpoint.y)), (int(self.target.center.x), int(self.target.center.y)),(255,0,0), 3)
        cv2.imwrite(f'results/best_shot_{self.path.split(".")[0].split("/")[1]}.png', img)
    def is_possible(self):
        if self.angle > 75:
            return False
        if not self.close_enough():
            return False
        if self.check_near():
            return False
        if self.check_collision_2():
            return False
        if self.check_collision_1():
            return False
        return True
        
    def check_collision_1(self):
        cue_points = self.balls_dist.perpendicular_line(self.cue.center).intersection(self.cue)
        parallel1 = self.balls_dist.parallel_line(cue_points[0])
        parallel2 = self.balls_dist.parallel_line(cue_points[1])
        p1 = parallel1.intersection(self.target)[0]
        p2 = parallel2.intersection(self.target)[0]
        line1 = Segment2D(cue_points[0], p1)
        line2 = Segment2D(cue_points[1], p2)
        for ball in self.balls:
            if ball == self.cue or ball == self.target:
                continue
            if line1.intersection(ball):
                return True
            if line2.intersection(ball):
                return True
        return False
    def check_collision_2(self):
        target_points = self.lusa_dist.perpendicular_line(self.target.center).intersection(self.target)
        if target_points[0].x <= target_points[1].x:
            line1 = Segment2D(target_points[0], self.lusa.points[0])
            line2 = Segment2D(target_points[1], self.lusa.points[1])
        else:
            line1 = Segment2D(target_points[1], self.lusa.points[0])
            line2 = Segment2D(target_points[0], self.lusa.points[1])
        for ball in self.balls:
            if ball == self.cue or ball == self.target:
                continue
            if line1.intersection(ball):
                return True
            if line2.intersection(ball):
                return True
        return False
    def close_enough(self):
        ray = Ray2D(self.target.center, self.cue.center)
        inter = ray.intersection(self.field)[0]
        dist = Segment2D(inter, self.cue.center).length
        if dist < 1000:
            return True
        return False
    def check_near(self):
        new_ball_cue = Circle(self.cue.center, 35)
        new_ball_target = Circle(self.target.center, 35)
        for ball in self.balls:
            if ball == self.cue or ball == self.target:
                continue
            if new_ball_cue.intersection(ball):
                return True
            if new_ball_target.intersection(ball):
                return True
        return False
    def calculate_score(self):
        self.score = (self.balls_dist.length - 60) * 1.2 + self.lusa_dist.length - 60 + self.angle * 10 + self.lusa_angle * 30
        
def avg(arr):
    new_arr = [a[1] for a in arr]
    return int(sum(new_arr)/len(new_arr))

def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    px = round(( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ))
    py = round(( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ))
    return [px, py]

class Balls_Detector():
    def __init__(self, path) -> None:
        self.raw = cv2.imread(path)
        self.path = path
    def find(self):
        self.transform()
        self.balls_detector()
        self.filter_cords()
        self.find_shot()

    def transform(self):
        img = self.raw

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        l_b = np.array([160,160,130])
        u_b = np.array([179,255,255])

        mask = cv2.inRange(hsv,l_b,u_b)
        res = cv2.bitwise_and(img,img,mask = mask)
        height, width = res.shape[:2]
        left = res[:, :width//2, :]
        right = res[:, width//2:, :]
        top_left = left[:height//2, :, :]
        bot_left = left[height//2:, :, :]
        top_right = right[:height//2, :, :]
        bot_right = right[height//2:, :, :]

        ans_top_left = []
        ans_bot_left = []
        ans_top_right = []
        ans_bot_right = []

        for i in range(height//2):
            for j in range(width//2):
                if (top_left[i, j, :] != np.array([0, 0, 0])).all():
                    ans_top_left.append((j, i))
                if (bot_left[i, j, :] != np.array([0, 0, 0])).all():
                    ans_bot_left.append((j, i))
                if (top_right[i, j, :] != np.array([0, 0, 0])).all():
                    ans_top_right.append((j, i))
                if (bot_right[i, j, :] != np.array([0, 0, 0])).all():
                    ans_bot_right.append((j, i)) 
        
        top_left_top_x = min([elem[0] for elem in ans_top_left if elem[1] <= avg(ans_top_left)])
        top_left_top_y = max([elem[1] for elem in ans_top_left if elem[1] <= avg(ans_top_left)])

        top_left_bot_x = max([elem[0] for elem in ans_top_left if elem[1] >= avg(ans_top_left)])
        top_left_bot_y = min([elem[1] for elem in ans_top_left if elem[1] >= avg(ans_top_left)])

        top_right_top_x = max([elem[0] for elem in ans_top_right if elem[1] <= avg(ans_top_right)]) + width//2
        top_right_top_y = max([elem[1] for elem in ans_top_right if elem[1] <= avg(ans_top_right)])

        top_right_bot_x = min([elem[0] for elem in ans_top_right if elem[1] >= avg(ans_top_right)]) + width//2
        top_right_bot_y = min([elem[1] for elem in ans_top_right if elem[1] >= avg(ans_top_right)]) 


        bot_right_top_x = min([elem[0] for elem in ans_bot_right if elem[1] <= avg(ans_bot_right)]) + width//2
        bot_right_top_y = max([elem[1] for elem in ans_bot_right if elem[1] <= avg(ans_bot_right)]) + height//2

        bot_right_bot_x = max([elem[0] for elem in ans_bot_right if elem[1] >= avg(ans_bot_right)]) + width//2
        bot_right_bot_y = min([elem[1] for elem in ans_bot_right if elem[1] >= avg(ans_bot_right)]) + height//2

        bot_left_top_x = max([elem[0] for elem in ans_bot_left if elem[1] <= avg(ans_bot_left)]) 
        bot_left_top_y = max([elem[1] for elem in ans_bot_left if elem[1] <= avg(ans_bot_left)]) + height//2

        bot_left_bot_x = min([elem[0] for elem in ans_bot_left if elem[1] >= avg(ans_bot_left)]) 
        bot_left_bot_y = min([elem[1] for elem in ans_bot_left if elem[1] >= avg(ans_bot_left)]) + height//2

        point1 = findIntersection(top_left_top_x, top_left_top_y, top_right_top_x, top_right_top_y, bot_left_top_x, bot_left_top_y, top_left_bot_x, top_left_bot_y)
        point2 = findIntersection(top_left_top_x, top_left_top_y, top_right_top_x, top_right_top_y, top_right_bot_x, top_right_bot_y, bot_right_top_x, bot_right_top_y)
        point3 = findIntersection(top_right_bot_x, top_right_bot_y, bot_right_top_x, bot_right_top_y, bot_right_bot_x, bot_right_bot_y, bot_left_bot_x, bot_left_bot_y)
        point4 = findIntersection(bot_right_bot_x, bot_right_bot_y, bot_left_bot_x, bot_left_bot_y, bot_left_top_x, bot_left_top_y, top_left_bot_x, top_left_bot_y)

        new_ans = np.float32([point1, point2, point3, point4])
        output = np.float32([[0, 0], [1270-1, 0], [1270-1, 2537-1], [0, 2537-1]])
        m = cv2.getPerspectiveTransform(new_ans, output)
        outimg = cv2.warpPerspective(img, m, (1270, 2537), cv2.INTER_LINEAR)
        cv2.imwrite('intermediate/top_view.png',outimg)
        self.top = outimg
        return outimg

    def mask(self,frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_b = np.array([0,0,100])
        u_b = np.array([100,95,255])
        mask = cv2.inRange(hsv,l_b,u_b)
        img = cv2.bitwise_and(frame,frame,mask = mask)
        return img

    def balls_detector(self):
        frame = self.top
        img = self.mask(frame)
        img = cv2.bilateralFilter(img, 20, 60, 60)
        kernel = np.ones((20,20),dtype=np.uint8)
        im1 = cv2.erode(img,kernel)
        gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('intermediate/gray.png',gray)
        edged = cv2.Canny(gray, 85, 255)
        cv2.imwrite('intermediate/canny.png',edged)
        contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        new_c = []
        for contour in contours:
            if 60 <= len(contour):
                new_c.append(contour)
        blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        cv2.drawContours(blank_image,new_c,-1, (0, 0, 255), 1)
        gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                  param1=30,
                  param2=15,
                  minRadius=8,
                  maxRadius=30)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(frame,(i[0],i[1]),30,(0,0,255),3)
        cv2.imwrite('intermediate/balls_found.png', frame)
        final = []
        circles = circles[0]
        for i in range(circles.shape[0]):
            final.append(Point2D(circles[i][0], circles[i][1]))
        self.cords = final
        return final


    def filter_cords(self):
        self.X = 1270
        self.Y = 2537
        self.R = 30
        self.lusy = [Segment2D(Point2D(55-1, 0), Point2D(0, 55-1)), Segment2D(Point2D(1215-1, 0), Point2D(1270-1, 55-1)),
                 Segment2D(Point2D(1270-1, 1230-1), Point2D(1270-1, 1307-1)), Segment2D(Point2D(1270-1, 2482-1), Point2D(1215-1, 2537-1)),
                   Segment2D(Point2D(0, 2482-1), Point2D(55-1, 2537-1)), Segment2D(Point2D(0, 1230-1), Point2D(0, 1307-1))]
        self.lusy_s =[Point2D(0, 0), Point2D(1270-1, 0), Point2D(1270-1, 1269-1), Point2D(1270-1, 2537-1), Point2D(0, 2537-1), Point2D(0, 1269-1)]
        self.field = Polygon(Point2D(0, 0), Point2D(1270-1, 0),Point2D(1270-1, 2537-1), Point2D(0, 2537-1))
        self.balls = []
        self.shots = []
        for point in self.cords:
            self.balls.append(Circle(point, self.R))
        for i, ball1 in enumerate(self.balls):
            for j, ball2 in enumerate(self.balls):
                if i != j:
                    for lusa in self.lusy:
                        shot = Shot(ball1, ball2, lusa, self.balls, self.field, self.path)
                        self.shots.append(shot)

    def find_shot(self):
        for shot in self.shots:
            shot.calculate_score()
        self.shots.sort(key=lambda x: x.score)
        for shot in self.shots:
            if shot.is_possible():
                shot.paint()
                return


det = Balls_Detector('photos/photo1.jpg')
det.find()

