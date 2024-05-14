import sdl2
import sdl2.ext

class Display(object):
    def __init__(self, W: int or float, H: int or float):
        sdl2.ext.init()

        self.W, self.H = W, H
        self.window = sdl2.ext.Window("SDL2 Window", size=(W, H), position=(500, 100))
        self.window.show()

    def paint(self, frame):
        # SDL2 boilerplate
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        # Draw on surface
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:, :, 0:3] = frame.swapaxes(0, 1)

        # Blit
        self.window.refresh()
