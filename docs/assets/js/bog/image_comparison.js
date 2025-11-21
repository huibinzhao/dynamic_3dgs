(function () {
    'use strict';

    const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

    function setupComparisonSlider(slider) {
        const input = slider.querySelector('.comparison-slider__input');
        if (!input) {
            return;
        }

        let isDragging = false;

        const activate = () => slider.classList.add('comparison-slider--active');
        const deactivate = () => slider.classList.remove('comparison-slider--active');

        const updatePosition = (value) => {
            const numericValue = typeof value === 'number' ? value : parseFloat(value);
            slider.style.setProperty('--position', numericValue);
        };

        const updateFromPointer = (clientX) => {
            const rect = slider.getBoundingClientRect();
            const raw = ((clientX - rect.left) / rect.width) * 100;
            const clamped = clamp(raw, 0, 100);
            input.value = clamped;
            updatePosition(clamped);
        };

        updatePosition(input.value || 50);

        input.addEventListener('input', (event) => {
            updatePosition(event.target.value);
        });

        slider.addEventListener('pointerdown', (event) => {
            isDragging = true;
            activate();
            updateFromPointer(event.clientX);

            const handleMove = (moveEvent) => {
                updateFromPointer(moveEvent.clientX);
            };

            const stopTracking = () => {
                isDragging = false;
                deactivate();
                window.removeEventListener('pointermove', handleMove);
                window.removeEventListener('pointerup', stopTracking);
            };

            window.addEventListener('pointermove', handleMove);
            window.addEventListener('pointerup', stopTracking, { once: true });
        });

        slider.addEventListener('pointermove', (event) => {
            if (!isDragging && event.pointerType === 'mouse' && event.buttons === 0) {
                activate();
                updateFromPointer(event.clientX);
            }
        });

        slider.addEventListener('pointerenter', () => {
            if (!isDragging) {
                activate();
            }
        });

        slider.addEventListener('pointerleave', () => {
            if (!isDragging) {
                deactivate();
            }
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('.comparison-slider').forEach(setupComparisonSlider);
    });
})();
