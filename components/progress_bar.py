import streamlit as st

class FrameCounter():
    # Use for counting elapsed frame in the live stream
    def __init__(self):
        self.frame = 0

    def __call__(self, count=0):
        self.frame += count
        return self.frame

class Progress_bar():
    def __init__(self, id, total_frame) -> None:
        # show analysis progress
        self.total_frame = total_frame
        self.id = id
        self.progress_txt = st.caption( f'Analysing Video: 0 out of {int( total_frame )} frames' )
        self.progress_bar = st.progress( 0 )
        self.counter = FrameCounter()

    def progress(self, iter=1):
        # progress of analysis
        self.progress_bar.progress( self.counter( iter ) / self.total_frame )
        self.progress_txt.caption( f'Analysing Video: {self.counter( 0 )} out of {self.total_frame} frames' )

    def terminate(self):
        self.progress_bar.progress( 100 )
        self.progress_txt.empty()