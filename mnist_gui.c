// mnist_gui.c – компактное Win32-GUI для MNIST
// gcc -std=c11 mnist_gui.c -lgdi32 -o mnist_gui.exe
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

// ---------- определение нейросети ---------- //
#define INPUT_SIZE     784 // 28*28 пикселей
#define OUTPUT_CLASSES 10 // классы цифр 0-9
#define MAX_LAYERS     10 // максимальное число слоев

typedef struct {
    int n; // число нейронов в слое
    double *a,*b,*z,**w; // a: активации, b: смещения, z: взвешенные суммы, w: веса
} Layer;

typedef struct {
    int L, sizes[MAX_LAYERS]; // число слоев и массив размеров слоев
    Layer layer[MAX_LAYERS];
} Net;

/**
 * @brief  Функция активации ReLU.
 *
 * Возвращает вход, если он положителен, иначе 0.
 *
 * @param  x  входное значение
 * @return double  `max(0, x)`
 */
static double relu(double x){ return x>0?x:0; }

/**
 * @brief  Softmax для последнего слоя.
 *
 * @param  z     вектор взвешенных сумм (длина n)
 * @param  out   выходной массив вероятностей (длина n)
 * @param  n     число классов
 */
static void softmax(const double*z,double*out,int n){
    double m=z[0]; for(int i=1;i<n;i++) if(z[i]>m) m=z[i];
    double s=0; for(int i=0;i<n;i++){ out[i]=exp(z[i]-m); s+=out[i]; }
    for(int i=0;i<n;i++) out[i]/=s;
}

/**
 * @brief  Создаёт слой нейронов с ReLU-инициализацией Хе.
 *
 * @param  n     количество нейронов в создаваемом слое
 * @param  prev  размер предыдущего слоя (0 — если входной)
 * @return Layer полностью инициализированная структура Layer
 */
static Layer make_layer(int n,int prev){
    Layer L={.n=n};
    L.a=calloc(n,sizeof*L.a); L.b=calloc(n,sizeof*L.b);
    L.z=calloc(n,sizeof*L.z); L.w=prev?malloc(n*sizeof*L.w):NULL;
    double sd=sqrt(2.0/prev);
    for(int i=0;i<n&&prev;i++){
        L.w[i]=malloc(prev*sizeof**L.w);
        for(int j=0;j<prev;j++) L.w[i][j]=sd*((double)rand()/RAND_MAX*2-1);
    } return L;
}


/**
 * @brief  Собирает полную сеть из массива размеров.
 *
 * @param  sizes массив размеров слоёв (длина L)
 * @param  L     количество слоёв
 * @return Net   инициализированная сеть
 */
static Net make_net(const int*sizes,int L){
    Net net={.L=L};
    for(int i=0;i<L;i++){ net.sizes[i]=sizes[i]; net.layer[i]=make_layer(sizes[i],i?sizes[i-1]:0); }
    return net;
}

/**
 * @brief  Загружает обученные веса/смещения из `weights.txt`.
 *
 * Формат файла соответствует сохранённому из консольной версии.
 *
 * @param  net   уже сконструированная сеть (make_net)
 * @param  file  путь к файлу весов
 *
 * @warning При несоответствии числа или размеров слоёв
 *          функция завершает программу через MessageBox.
 */
static void load_weights(Net*net,const char*file){
    FILE*fp=fopen(file,"r");
    if(!fp){ MessageBoxA(NULL,"Не найден weights.txt","Error",MB_ICONERROR); exit(1);}
    int L; fscanf(fp,"%d",&L); if(L!=net->L){ MessageBoxA(NULL,"layer mismatch","Error",MB_ICONERROR); exit(1);}
    for(int i=0;i<L;i++){ int tmp; fscanf(fp,"%d",&tmp);} double lr,reg; fscanf(fp,"%lf%lf",&lr,&reg);
    // для каждого слоя считываем b[i] и w[i][j]
    for(int l=1;l<L;l++){ Layer*c=&net->layer[l]; for(int i=0;i<c->n;i++)fscanf(fp,"%lf",&c->b[i]);
        int prev=net->sizes[l-1]; for(int i=0;i<c->n;i++)for(int j=0;j<prev;j++)fscanf(fp,"%lf",&c->w[i][j]);
    } fclose(fp);
}

/**
 * @brief  Прямой проход по всей сети.
 *
 * @param  net   сеть
 * @param  in    входное изображение 28×28, развёрнутое в вектор 784
 * @param  out   массив для результатов softmax (длина OUTPUT_CLASSES)
 */
static void forward(Net*net,const double*in,double*out){
    memcpy(net->layer[0].a,in,net->sizes[0]*sizeof(double));
    for(int l=1;l<net->L;l++){
        Layer*p=&net->layer[l-1],*c=&net->layer[l];
        for(int i=0;i<c->n;i++){ double sum=c->b[i]; for(int j=0;j<p->n;j++) sum+=c->w[i][j]*p->a[j];
            c->z[i]=sum; c->a[i]=(l==net->L-1)?sum:relu(sum);}
        if(l==net->L-1) softmax(c->z,c->a,c->n);
    }
    memcpy(out,net->layer[net->L-1].a,OUTPUT_CLASSES*sizeof(double));
}

/* ---------- GUI ---------- */
#define CANVAS 280 // размер канвы
#define PEN_W  12 // толщина кисти
static HBITMAP hDib=NULL; static void*dibPix=NULL; static HDC hMemDC=NULL; // dib для рисования
static bool drawing=false; // флаг "рисую ли мышью"
static POINT lastPt; // последняя точка при рисовании
static double probs[OUTPUT_CLASSES]={0}; // вероятности для цифр 0–9
static Net net;

/**
 * @brief  Пересчитывает вероятность цифр исходя из содержимого Canvas.
 *
 * 1. Преобразует 280×280 пикселей холста в 28×28 ячеек  
 *    (усреднение яркости внутри каждой ячейки).  
 * 2. Нормирует значения в диапазон [0;1].  
 * 3. Запускает @ref forward() и кладёт результат в `probs`.
 */
static void infer(){
    double in[INPUT_SIZE];
    int stride=CANVAS*3; // 3 байта на пиксель
    for(int cy=0; cy<28; cy++){
        for(int cx=0; cx<28; cx++){
            int sum=0;
            for(int y=0; y<10; y++){
                uint8_t*row=(uint8_t*)dibPix + (cy*10+y)*stride + cx*10*3;
                for(int x=0; x<10; x++) sum += row[x*3];
            }
            in[cy*28+cx] = 1.0 - (sum/100.0)/255.0;
        }
    }
    forward(&net,in,probs);
}

/**
 * @brief  Полностью очищает холст и сбрасывает предсказание.
 *
 * Заливает буфер белым, вызывает @ref infer() и
 * инициирует перерисовку окна.
 */
static void clear_canvas(){
    // заливаем все белым
    memset(dibPix,255,CANVAS*CANVAS*3);
    // пересчитываем предсказание на уже пустом холсте
    infer();
    // просим windows перерисовать всё окно (TRUE — очистить фон полностью)
    InvalidateRect(NULL,NULL,TRUE);
}

/**
 * @brief  Рисует толстую (PEN_W) линию на DIB-холсте.
 *
 * @param  from  начальная точка (координаты в окне)
 * @param  to    конечная точка
 */
static void draw_line(POINT from,POINT to){
    // cоздаём перо толщиной PEN_W и цветом RGB(0,0,0) (чёрное)
    HPEN hPen=CreatePen(PS_SOLID,PEN_W,RGB(0,0,0));
    HGDIOBJ old=SelectObject(hMemDC,hPen);
    // перемещаем «перо» в точку from и рисуем линию до to
    MoveToEx(hMemDC,from.x,from.y,NULL); LineTo(hMemDC,to.x,to.y);
    SelectObject(hMemDC,old); DeleteObject(hPen);
}

/**
 * @brief  Главная обработка сообщений Win32.
 *
 * Реагирует на:
 * * создание окна — инициализация DIB-буфера;  
 * * сообщения мыши — рисование и запуск @ref infer();  
 * * правый клик или клавишу C — очистка полотна;  
 * * WM_PAINT — отрисовка холста и панельки вероятностей.
 *
 * @param  h     дескриптор окна
 * @param  msg   код сообщения
 * @param  w     WPARAM
 * @param  l     LPARAM
 * @return LRESULT  стандартно для Win32
 */
static LRESULT CALLBACK WndProc(HWND h,UINT msg,WPARAM w,LPARAM l){
    switch(msg){
        case WM_CREATE:{
            BITMAPINFO bmi={0};
            bmi.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth=CANVAS; bmi.bmiHeader.biHeight=-CANVAS;
            bmi.bmiHeader.biPlanes=1; bmi.bmiHeader.biBitCount=24; bmi.bmiHeader.biCompression=BI_RGB;
            hDib=CreateDIBSection(NULL,&bmi,DIB_RGB_COLORS,&dibPix,NULL,0);
            memset(dibPix,255,CANVAS*CANVAS*3);
            hMemDC=CreateCompatibleDC(NULL); SelectObject(hMemDC,hDib);
            infer(); break;}
        case WM_LBUTTONDOWN: drawing=true; lastPt.x=LOWORD(l); lastPt.y=HIWORD(l); break;
        case WM_MOUSEMOVE:   if(drawing){ POINT pt={LOWORD(l),HIWORD(l)};
                                    draw_line(lastPt,pt); lastPt=pt; infer(); InvalidateRect(h,NULL,FALSE);} break;
        case WM_LBUTTONUP:   drawing=false; infer(); InvalidateRect(h,NULL,FALSE); break;
        case WM_RBUTTONDOWN: clear_canvas(); break;
        case WM_KEYDOWN:     if(w=='C'||w=='c') clear_canvas(); break;

        case WM_PAINT:{
            PAINTSTRUCT ps; HDC dc=BeginPaint(h,&ps);
            /* холст */
            BitBlt(dc,10,10,CANVAS,CANVAS,hMemDC,0,0,SRCCOPY);
            RECT canvasR={10,10,10+CANVAS,10+CANVAS};
            FrameRect(dc,&canvasR,(HBRUSH)GetStockObject(BLACK_BRUSH));

            /* панель вероятностей */
            RECT panel={CANVAS+20,10,600-10,10+CANVAS};
            FillRect(dc,&panel,(HBRUSH)GetStockObject(WHITE_BRUSH));

            int xText=panel.left+5, y=panel.top+5;
            int barMax=220;
            for(int i=0;i<10;i++){
                /* строка текста */
                char buf[32]; sprintf(buf,"%d : %.3f",i,probs[i]);
                TextOutA(dc,xText,y,buf,strlen(buf));

                /* полоса */
                int len=(int)(probs[i]*barMax+0.5);
                HBRUSH bar=CreateSolidBrush(RGB(100,149,237)); // cornflower blue
                RECT barR={xText+80,y+2,xText+80+len,y+14};
                FillRect(dc,&barR,bar); DeleteObject(bar);

                y+=20;
            }
            EndPaint(h,&ps); break;}
        case WM_DESTROY: PostQuitMessage(0); break;
        default: return DefWindowProc(h,msg,w,l);
    }
    return 0;
}

/**
 * @brief  Точка входа GUI-приложения.
 *
 * 1. Читает `config.txt`, строит сеть и подгружает веса  
 *    (можно указать альтернативный файл через аргумент командной строки).  
 * 2. Регистрирует класс окна и запускает цикл сообщений.  
 * 3. Пользователь может рисовать цифры, а программа
 *    в режиме реального времени выводит вероятности 0-9.
 *
 * @return int код завершения, ожидаемый Windows.
 */
int WINAPI WinMain(HINSTANCE hi,HINSTANCE,LPSTR cmd,int){
    /* читаем config.txt */
    int sizes[MAX_LAYERS],L=0; FILE*cf=fopen("config.txt","r");
    if(!cf){ MessageBoxA(NULL,"config.txt not found","Error",MB_ICONERROR); return 1;}
    char ln[128]; while(fgets(ln,sizeof ln,cf))
        if(!strncmp(ln,"neurons:",8))
            for(char*p=strtok(ln+8,", \n"); p; p=strtok(NULL,", \n")) sizes[L++]=atoi(p);
    fclose(cf); if(L<2){ MessageBoxA(NULL,"config.txt must define ≥2 layers","Error",MB_ICONERROR); return 1;}

    net=make_net(sizes,L); load_weights(&net,*cmd?cmd:"weights.txt");

    WNDCLASSA wc={.lpfnWndProc=WndProc,.hInstance=hi,.hCursor=LoadCursor(NULL,IDC_ARROW),
                  .lpszClassName="mnistGUI",.hbrBackground=(HBRUSH)(COLOR_WINDOW+1)};
    RegisterClassA(&wc);
    HWND h=CreateWindowA("mnistGUI","MNIST GUI – нарисуйте цифру",
                         WS_OVERLAPPED|WS_CAPTION|WS_SYSMENU|WS_MINIMIZEBOX,
                         CW_USEDEFAULT,CW_USEDEFAULT,600,340,NULL,NULL,hi,NULL);
    ShowWindow(h,SW_SHOWDEFAULT);

    MSG m; while(GetMessage(&m,NULL,0,0)){ TranslateMessage(&m); DispatchMessage(&m);}
    return 0;
}
