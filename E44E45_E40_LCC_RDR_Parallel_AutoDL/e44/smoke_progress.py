"""Bounded real-batch smoke; AMP skips are attempts, NOT optimizer updates."""
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import shutil

MAX_ATTEMPTS = 32
TARGET_UPDATES = 8


@dataclass
class SmokeProgress:
    attempts: int = 0
    successful: int = 0
    skipped: int = 0

    def observe(self, update):
        self.attempts += 1
        if update['updated']:
            if update['status'] != 'finite_update' or update['bad_parameter_count'] != 0:
                raise RuntimeError('Invalid finite-update evidence')
            self.successful += 1
        else:
            if update['status'] != 'overflow_skip':
                raise RuntimeError('Skipped update without recorded AMP overflow')
            self.skipped += 1

    @property
    def ready(self):
        return self.successful >= TARGET_UPDATES

    def require_success(self):
        if not self.ready:
            raise RuntimeError(f'Smoke stopped after {self.attempts}/{MAX_ATTEMPTS} real batches: '
                               f'{self.successful}/{TARGET_UPDATES} successful optimizer updates, '
                               f'{self.skipped} AMP skips. No smoke receipt issued.')


def stress_batches(loader):
    return enumerate(islice(loader, MAX_ATTEMPTS))


def seed_existing_font(project, settings):
    """Reuse a genuine local font; never fabricate one or modify shared source."""
    destination = Path(settings) / 'Arial.ttf'
    if destination.is_file():
        return
    for name in ('e40_settings', 'e41_settings', 'e40_seed2_settings', 'e41_seed2_settings', 'e42_settings', 'e43_settings'):
        source = Path(project) / 'comparison_reports' / name / 'Arial.ttf'
        if source.is_file() and source.stat().st_size > 100_000:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.copying')
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            print('Reused existing Arial.ttf: ' + str(source), flush=True)
            return
