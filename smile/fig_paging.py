"""
Matplotlib figure paging functions used by the processing pipeline for
https://github.com/mcmhsieh/Smile to accommodate multiple figures within
each figure window that can be paged in and out by clicking button widgets
added to the figure toolbar.

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt


subplotpars_keys = ['left', 'bottom', 'right', 'top', 'wspace', 'hspace']
fig_subplotpars = {}
fig_pages = {}
fig_toolbar_nav_stacks = {}
fig_toolbar_widgets = {}
fig_current_page_idxs = {}

def get_fig_children(fig):
    children = []
    for child in fig.get_children():
        if child is not fig.patch:
            child_axes = child.axes if hasattr(child, 'axes') else None
            # Persist shared axes
            # TODO: persist twinned axes
            axes_siblings = {}
            if isinstance(child, matplotlib.axes.Axes):
                for name in child._axis_names:
                    grouper = child._shared_axes[name]
                    axes_siblings[name] = [other for other in grouper.get_siblings(child) if other is not child]
            children.append((child, child_axes, axes_siblings))
    return children

def clear_fig(fig):
    for child, _, _ in get_fig_children(fig):
        # Call remove() prior to Figure.clear() to avoid axes being cleared
        child.remove()
        if isinstance(child, matplotlib.axes.Axes):
            # Restore reference to figure, which may still be used by e.g. event callback handlers
            child.figure = fig
    # Call Figure.clear() to remove references to internal labels such as Figure._suptitle
    fig.clear()
    # Reset the subplots grid positioning parameters
    if fig.number in fig_subplotpars:
        fig.subplotpars.update(**fig_subplotpars[fig.number])

def save_toolbar_nav_stack(fig):
    if fig.number in fig_current_page_idxs:
        # Save the view navigation history
        current_page_idx = fig_current_page_idxs[fig.number]
        toolbar_nav_stack = fig.canvas.manager.toolbar._nav_stack
        fig_toolbar_nav_stacks[fig.number][current_page_idx] = (toolbar_nav_stack(),
                                                                [toolbar_nav_stack[idx] for idx in range(len(toolbar_nav_stack))])

def update_fig_page_idx(fig, page_idx):
    fig_current_page_idxs[fig.number] = page_idx
    page_label, _, _, _, _ = fig_toolbar_widgets[fig.number]
    page_label.setText(f'{page_idx + 1}/{len(fig_pages[fig.number])}')

def show_fig_page(fig, page_idx):
    if page_idx is None:
        return

    save_toolbar_nav_stack(fig)

    update_fig_page_idx(fig, page_idx)

    clear_fig(fig)

    for child, child_axes, axes_siblings in fig_pages[fig.number][page_idx]:
        if hasattr(child, 'axes'):
            child.axes = child_axes
        if isinstance(child, matplotlib.axes.Axes):
            # Note that sharex and sharey keywords are only applicable when creating a new axes,
            # not when a pre-existing axes is being added.
            fig.add_axes(child)
            # Restore shared axes
            # TODO: restore twinned axes
            for name in child._axis_names:
                # _AxesBase retains separate references to the shared axes that were originally set
                # despite the share links being broken when the remove() method is called.
                shared_axes = getattr(child, f'_share{name}')
                if shared_axes is not None:
                    getattr(child, f'share{name}')(shared_axes)
                else:
                    for ax in fig.get_axes():
                        if ax in axes_siblings[name]:
                            getattr(child, f'share{name}')(ax)
                            break
        else:
            fig.add_artist(child)

    toolbar = fig.canvas.manager.toolbar
    toolbar_nav_stack = toolbar._nav_stack
    assert len(toolbar_nav_stack) == 0
    current_element, stack_elements = fig_toolbar_nav_stacks[fig.number][page_idx]
    if current_element is not None:
        for element in stack_elements:
            toolbar_nav_stack.push(element)
        while toolbar_nav_stack() is not current_element:
            toolbar_nav_stack.back()
        toolbar.set_history_buttons()
        toolbar._update_view()
    else:
        fig.canvas.draw_idle()

def setup_new_fig_page():
    fig = plt.gcf()

    if fig.number not in fig_subplotpars:
        fig_subplotpars[fig.number] = {key: getattr(fig.subplotpars, key) for key in subplotpars_keys}

    save_toolbar_nav_stack(fig)

    clear_fig(fig)

    if fig.number in fig_pages:
        return

    fig_pages[fig.number] = []
    fig_toolbar_nav_stacks[fig.number] = {}

    toolbar = fig.canvas.manager.toolbar

    page_label = matplotlib.backends.qt_compat.QtWidgets.QLabel()
    prev_page_action = matplotlib.backends.qt_compat.QtWidgets.QAction(toolbar._icon('back_large.png'), 'Previous Page')
    next_page_action = matplotlib.backends.qt_compat.QtWidgets.QAction(toolbar._icon('forward_large.png'), 'Next Page')
    back_10_page_action = matplotlib.backends.qt_compat.QtWidgets.QAction('«㉈')
    forward_10_page_action = matplotlib.backends.qt_compat.QtWidgets.QAction('»㉈')

    font = matplotlib.backends.qt_compat.QtGui.QFont()
    font.setPixelSize(20)
    back_10_page_action.setFont(font)
    forward_10_page_action.setFont(font)

    toolbar.insertSeparator(toolbar.actions()[-1])
    toolbar.insertAction(toolbar.actions()[-1], back_10_page_action)
    toolbar.insertAction(toolbar.actions()[-1], prev_page_action)
    toolbar.insertWidget(toolbar.actions()[-1], page_label)
    toolbar.insertAction(toolbar.actions()[-1], next_page_action)
    toolbar.insertAction(toolbar.actions()[-1], forward_10_page_action)
    toolbar.insertSeparator(toolbar.actions()[-1])

    # Maintain references to Qt widgets to prevent them from being GC'd
    fig_toolbar_widgets[fig.number] = (page_label, prev_page_action, next_page_action, back_10_page_action, forward_10_page_action)

    prev_page_action.triggered.connect(lambda: show_fig_page(fig, (fig_current_page_idxs[fig.number] - 1) % len(fig_pages[fig.number])
                                                             if fig.number in fig_current_page_idxs else None))
    next_page_action.triggered.connect(lambda: show_fig_page(fig, (fig_current_page_idxs[fig.number] + 1) % len(fig_pages[fig.number])
                                                             if fig.number in fig_current_page_idxs else None))
    back_10_page_action.triggered.connect(lambda: show_fig_page(fig, (fig_current_page_idxs[fig.number] - 10) % len(fig_pages[fig.number])
                                                                if fig.number in fig_current_page_idxs else None))
    forward_10_page_action.triggered.connect(lambda: show_fig_page(fig, (fig_current_page_idxs[fig.number] + 10) % len(fig_pages[fig.number])
                                                                   if fig.number in fig_current_page_idxs else None))

    def on_close(event):
        if fig.number in fig_subplotpars:
            del fig_subplotpars[fig.number]
        if fig.number in fig_pages:
            del fig_pages[fig.number]
        if fig.number in fig_toolbar_nav_stacks:
            del fig_toolbar_nav_stacks[fig.number]
        if fig.number in fig_toolbar_widgets:
            del fig_toolbar_widgets[fig.number]
        if fig.number in fig_current_page_idxs:
            del fig_current_page_idxs[fig.number]
    fig.canvas.mpl_connect('close_event', on_close)

def stash_fig_page():
    fig = plt.gcf()
    fig_pages[fig.number].append(get_fig_children(fig))

    page_idx = len(fig_pages[fig.number]) - 1
    update_fig_page_idx(fig, page_idx)


if __name__ == '__main__':

    plt.close('test1')
    plt.close('test2')

    plt.figure('test1', figsize=(16, 10))
    setup_new_fig_page()
    plt.suptitle('1')
    ax = plt.subplot(1, 3, 1)
    plt.plot(np.arange(10), np.arange(10))
    ax = plt.subplot(1, 3, 2, sharex=ax, sharey=ax)
    plt.plot(np.arange(10), np.arange(10))
    ax = plt.subplot(1, 3, 3, projection='3d')
    plt.scatter(np.arange(10), np.arange(10), np.arange(10))
    plt.tight_layout()
    stash_fig_page()

    plt.figure('test1', figsize=(16, 10))
    setup_new_fig_page()
    plt.suptitle('2')
    plt.plot(np.arange(10), 2 * np.arange(10))
    plt.tight_layout()
    stash_fig_page()

    plt.figure('test2', figsize=(16, 10))
    setup_new_fig_page()

    plt.figure('test1', figsize=(16, 10))
    setup_new_fig_page()
    plt.suptitle('3')
    plt.plot(np.arange(10), 3 * np.arange(10))
    plt.tight_layout()
    stash_fig_page()
