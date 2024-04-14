import tkinter as tk
import numpy as np
class Shadow(tk.Tk):

    def __init__(self, widget, color='#212121', size=5, offset_x=0, offset_y=0,
                 onhover={}, onclick={}):
        self.widget = widget
        self.normal_size = size
        self.normal_color = color
        self.normal_x = int(offset_x)
        self.normal_y = int(offset_y)
        self.onhover_size = onhover.get('size', size)
        self.onhover_color = onhover.get('color', color)
        self.onhover_x = onhover.get('offset_x', offset_x)
        self.onhover_y = onhover.get('offset_y', offset_y)
        self.onclick_size = onclick.get('size', size)
        self.onclick_color = onclick.get('color', color)
        self.onclick_x = onclick.get('offset_x', offset_x)
        self.onclick_y = onclick.get('offset_y', offset_y)
        
        # Get master and master's background
        self.master = widget.master
        self.to_rgb = tuple([el//257 for el in self.master.winfo_rgb(self.master.cget('bg'))])
        
        # Start with normal view
        self.__lines = []
        self.__normal()
        
        # Bind events to widget
        self.widget.bind("<Enter>", self.__onhover, add='+')
        self.widget.bind("<Leave>", self.__normal, add='+')
        self.widget.bind("<ButtonPress-1>", self.__onclick, add='+')
        self.widget.bind("<ButtonRelease-1>", self.__normal, add='+')
    
    def __normal(self, event=None):
        ''' Update shadow to normal state '''
        self.shadow_size = self.normal_size
        self.shadow_color = self.normal_color
        self.shadow_x = self.normal_x
        self.shadow_y = self.normal_y
        self.display()
    
    def __onhover(self, event=None):
        ''' Update shadow to hovering state '''
        self.shadow_size = self.onhover_size
        self.shadow_color = self.onhover_color
        self.shadow_x = self.onhover_x
        self.shadow_y = self.onhover_y
        self.display()
    
    def __onclick(self, event=None):
        ''' Update shadow to clicked state '''
        self.shadow_size = self.onclick_size
        self.shadow_color = self.onclick_color
        self.shadow_x = self.onclick_x
        self.shadow_y = self.onclick_y
        self.display()
    
    def __destroy_lines(self):
        
        for ll in self.__lines:
            ll.destroy()
        self.__lines = []
    
    def display(self):
        
        def _rgb2hex(rgb):
           
            return "#%02x%02x%02x" % rgb
    
        def _hex2rgb(h):
               
                h = h.strip('#')
                return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        
        # Destroy old lines
        self.__destroy_lines()
        
        # Get widget position and size
        self.master.update_idletasks()
        x0, y0, w, h = self.widget.winfo_x(), self.widget.winfo_y(), self.widget.winfo_width(), self.widget.winfo_height()
        x1 = x0 + w - 1
        y1 = y0 + h - 1
        
        # Get shadow size from borders
        if type(self.shadow_size) is int:
            wh_shadow_size = self.shadow_size
        else:
            wh_shadow_size = min([int(dim * (self.shadow_size - 1)) for dim in (w,h)])
        uldr_shadow_size = wh_shadow_size - self.shadow_y, wh_shadow_size - self.shadow_x, \
                           wh_shadow_size + self.shadow_y, wh_shadow_size + self.shadow_x
        uldr_shadow_size = {k:v for k,v in zip('uldr', uldr_shadow_size)}
        self.uldr_shadow_size = uldr_shadow_size
        
        # Prepare shadow color
        shadow_color = self.shadow_color
        if not shadow_color.startswith('#'):
            shadow_color = _rgb2hex(tuple([min(max(self.to_rgb) + 30, 255)] * 3))
        self.from_rgb = _hex2rgb(shadow_color)
        
        # Draw shadow lines
        max_size = max(uldr_shadow_size.values())
        diff_size = {k: max_size-ss for k,ss in uldr_shadow_size.items()}
        rs = np.linspace(self.from_rgb[0], self.to_rgb[0], max_size, dtype=int)
        gs = np.linspace(self.from_rgb[2], self.to_rgb[2], max_size, dtype=int)
        bs = np.linspace(self.from_rgb[1], self.to_rgb[1], max_size, dtype=int)
        rgbs = [_rgb2hex((r,g,b)) for r,g,b in zip(rs,gs,bs)]
        for direction, size in uldr_shadow_size.items():
            for ii, rgb in enumerate(rgbs):
                ff = tk.Frame(self.master, bg=rgb)
                self.__lines.append(ff)
                if direction=='u' or direction=='d':
                    diff_1 = diff_size['l']
                    diff_2 = diff_size['r']
                    yy = y0-ii+1+diff_size[direction] if direction == 'u' else y1+ii-diff_size[direction]
                    if diff_1 <= ii < diff_size[direction]:
                        ff1 = tk.Frame(self.master, bg=rgb)
                        self.__lines.append(ff1)
                        ff1.configure(width=ii+1-diff_1, height=1)
                        ff1.place(x=x0-ii+1+diff_size['l'], y=yy)
                    if diff_2 <= ii < diff_size[direction]:
                        ff2 = tk.Frame(self.master, bg=rgb)
                        self.__lines.append(ff2)
                        ff2.configure(width=ii+1-diff_2, height=1)
                        ff2.place(x=x1, y=yy)
                    if ii >= diff_size[direction]:
                        ff.configure(width=x1-x0+ii*2-diff_size['l']-diff_size['r'], height=1)
                        ff.place(x=x0-ii+1+diff_size['l'], y=yy)
                elif direction=='l' or direction=='r':
                    diff_1 = diff_size['u']
                    diff_2 = diff_size['d']
                    xx = x0-ii+1+diff_size[direction] if direction == 'l' else x1+ii-diff_size[direction]
                    if diff_1 <= ii < diff_size[direction]:
                        ff1 = tk.Frame(self.master, bg=rgb)
                        self.__lines.append(ff1)
                        ff1.configure(width=1, height=ii+1-diff_1)
                        ff1.place(x=xx, y=y0-ii+1+diff_size['u'])
                    if diff_2 <= ii < diff_size[direction]:
                        ff2 = tk.Frame(self.master, bg=rgb)
                        self.__lines.append(ff2)
                        ff2.configure(width=1, height=ii+1-diff_2)
                        ff2.place(x=xx, y=y1)
                    if ii >= diff_size[direction]:
                        ff.configure(width=1, height=y1-y0+ii*2-diff_size['u']-diff_size['d'])
                        ff.place(x=xx, y=y0-ii+1+diff_size['u'])