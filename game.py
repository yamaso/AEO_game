#!/usr/bin/python
import pygame
from pygame.locals import *
import sys
import os
import pyaudio
import numpy as np
import struct
import time
import python_speech_features
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# screen
SCR_RECT = Rect(0, 0, 1000, 750)
# サンプリング周波数
fs = 16000
# 録音トリガの振幅
lv = 5000
# 録音開始時刻フラグ
rec_start_time = ''
# 録音バッファ
data = []
# 認識クラスラベル
label_list_r = {0: "a", 1: "o", 2: "e"}


# 音声認識用ネットワーク定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(26, 100)  # 入力層, ユニット数26
        self.fc2 = nn.Linear(100, 50)  # 中間層, ユニット数100, 50
        self.fc3 = nn.Linear(50, 4)  # 出力層, ユニット数4

    def forward(self, x):  # 伝播（活性化関数）の定義
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MySprite(pygame.sprite.Sprite):
    JUMP_SPEED = 8.0  # ジャンプの初速度
    GRAVITY = 0.1  # 重力加速度

    # del_time = 120

    def __init__(self, filename, x, y, vx, vy):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(filename).convert_alpha()
        width = self.image.get_width()
        height = self.image.get_height()
        self.rect = Rect(x, y, width, height)
        self.vx = vx
        self.vy = vy
        self.on_floor = False

    def update(self):
        self.rect.move_ip(self.vx, self.vy)
        # 画面からはみ出ないようにする
        self.rect = self.rect.clamp(SCR_RECT)
        # キー入力取得
        pygame.event.pump()  # おまじない
        pressed_keys = pygame.key.get_pressed()
        # ジャンプ
        if pressed_keys[K_UP]:
            # if self.on_floor:
            self.vy = - self.JUMP_SPEED  # 上向きに初速度を与える
            self.on_floor = False

        if pressed_keys[K_RIGHT]:
            self.vx = + 3
        if pressed_keys[K_LEFT]:
            self.vx = - 3

        # 速度を更新
        if not self.on_floor:
            self.vy += self.GRAVITY  # 下向きに重力をかける

        # 浮動小数点の位置を更新
        # self.vx += self.vx
        # self.vy += self.vy

        # 着地したか調べる
        if self.vy > SCR_RECT.height - self.rect.height:
            # 床にめり込まないように位置調整
            self.vy = SCR_RECT.height - self.rect.height
            self.vy = 0
            self.on_floor = True

        # 浮動小数点の位置を整数座標に戻す
        # スプライトを動かすにはself.rectの更新が必要！
        # self.rect.x = int(self.vx)
        # self.rect.y = int(self.vy)
        # self.del_time -= 1
        # if self.del_time < 0:
        #     self.image.set_alpha(255)

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    # 音声認識の入力があった時
    def onrecog(self, text):
        if text == "o":
            # if self.on_floor:
            self.vy = - self.JUMP_SPEED  # 上向きに初速度を与える
            self.on_floor = False
        elif text == "e":
            self.vx = - 3  # xの変化量を-1にする
        elif text == "a":
            self.vx = + 3  # xの変化量を+1にする


# 録音用コーバック関数（別スレッドで動作）
def adCallback(in_data, frame_count, time_info, status):
    global rec_start_time

    # 取得した録音バッファ（in_data）をint16に変換
    buf = np.frombuffer(in_data, dtype="int16")
    # buf中の振幅の絶対値が lv を超えたら
    if np.max(np.abs(buf)) > lv:
        rec_start_time = time.time()

    # dataに録音バッファ（in_data）を追加
    if rec_start_time and not rec_start_time == "recog_start":
        # 振幅値がlvを超えて0.01秒以内なら継続
        if time.time() - rec_start_time <= 0.01:
            print(".", end="")
            data.append(in_data)

        # 振幅値がlvを下回って0.2秒たったら終了
        else:
            rec_start_time = "recog_start"

    return (None, pyaudio.paContinue)

# ここからメイン
if __name__ == '__main__':
    item_num = np.random.randint(3, 7)
    st = time.time()
    et = None
    prt_score = ""
    score = 0
    pygame.init()

    screen = pygame.display.set_mode(SCR_RECT.size)
    pygame.display.set_caption("pygame sample")

    font = pygame.font.Font(None, 55)

    pygame.mixer.music.load("moon.mp3")
    pygame.mixer.music.play(-1)
    get_sound = pygame.mixer.Sound("get1.wav")  # サウンドをロード
    get_sound2 = pygame.mixer.Sound("get2.wav")

    # 背景画像
    backImg = pygame.image.load("image/back.png").convert()

    # Spriteオブジェクトを作成
    jiki = MySprite("image/jiki1.png", 320, 240, 0, 0)
    items = [MySprite("image/bullet.png", np.random.randint(100, 900), np.random.randint(100, 650), 0, 0) for i in
             range(item_num)]

    # 音声認識のモデルをロード
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 以前の演習で作成したモデルをロードします。ファイルがないとエラーになります。
    model.load_state_dict(torch.load('model-aoe.mlp'))
    optimizer.load_state_dict(torch.load('model-aoe.optimizer'))

    audio = pyaudio.PyAudio()
    # stream for recording
    stream_ad = audio.open(format=pyaudio.paInt16,
                           channels=1,
                           rate=int(fs),
                           input=True,  # 録音のときは input=True
                           frames_per_buffer=2 ** 10,
                           stream_callback=adCallback)

    # main loop
    clock = pygame.time.Clock()
    item_flags = [True] * item_num
    end_flag = True
    while True:
        # 60fps
        clock.tick(60)

        # 背景を描画
        screen.blit(backImg, (0, 0))

        # Spriteオブジェクトの更新,描画
        jiki.update()
        jiki.draw(screen)
        for i in range(item_num):
            if items[i].rect.x > jiki.rect.left and items[i].rect.x < jiki.rect.right and items[
                i].rect.y > jiki.rect.top and items[i].rect.y < jiki.rect.bottom:
                item_flags[i] = False
                if end_flag:
                    get_sound.stop()
                    get_sound.play()  # サウンドを再生
            if item_flags[i]:
                items[i].draw(screen)
        if not end_flag:
            get_sound2.play()
        if set(item_flags) == {False} and end_flag:
            et = time.time()
            score = et - st
            end_flag = False
            pygame.mixer.music.fadeout(3000)

        if rec_start_time == "recog_start":
            print("recognition start.")
            data_mfcc = python_speech_features.mfcc(np.frombuffer(np.array(data), dtype="int16"), samplerate=fs)
            data_mfcc = data_mfcc / np.abs(data_mfcc.max())
            data_mfcc_ave = np.mean(data_mfcc[:], axis=0)
            data_mfcc_std = np.std(data_mfcc[:], axis=0)
            data_mfcc_con = np.concatenate((data_mfcc_ave, data_mfcc_std), axis=0)
            output = model(Variable(torch.from_numpy(data_mfcc_con).float()))
            conf = output.data.numpy()
            thr = [0.75, 0.33, 0.7]
            if conf.max() > thr[np.argmax(conf)]:
                results = np.argmax(conf)
                print(conf)
                print("result_text: " + label_list_r[results])

                # Spriteオブジェクトに認識結果を渡す
                jiki.onrecog(label_list_r[results])
            else:
                print(conf)
                print("result_text: " + "none")
            # 音声認識関係の変数を初期化
            data = []
            rec_start_time = ''
        if score:
            prt_score = str(100 - int(score)) + "point"
        if score < 100:
            text = font.render(str(prt_score), True, (255, 155, 255))
        else:
            text = font.render("Too Late! Hurry up!", True, (255, 155, 145))
        screen.blit(text, list(SCR_RECT.center))
        # 画面を更新
        pygame.display.update()

        # イベントハンドラ
        for event in pygame.event.get():
            if event.type == QUIT:
                # 終了処理
                audio.terminate()
                sys.exit()
