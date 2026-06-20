use game_engine_derive::Component;
use game_engine_ecs::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Component)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Component)]
struct Velocity {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Component)]
struct Health(f32);

#[derive(Clone, Copy, Debug, PartialEq, Component)]
struct Player;

#[derive(Clone, Copy, Debug, PartialEq, Component)]
struct Enemy;

#[derive(Default)]
struct GameState {
    frames: u32,
    score: u32,
}

#[derive(Default)]
struct ThreadMetrics {
    operations: u64,
}

struct CollisionMessage {
    entity_a: Entity,
    entity_b: Entity,
}

struct DestroyEvent {
    reason: u32,
}

fn spawn_system(commands: Commands, mut state: ResMut<GameState>) {
    state.frames += 1;
    commands.spawn((
        Position { x: 0.0, y: 0.0 },
        Velocity { x: 1.0, y: 1.0 },
        Player,
        Health(100.0),
    ));
    commands.spawn((Position { x: 50.0, y: 50.0 }, Enemy, Health(50.0)));
}

fn movement_system(
    mut query: Query<(&mut Position, &Velocity)>,
    mut local_timer: Local<u32>,
    mut tl_metrics: TLResMut<ThreadMetrics>,
) {
    *local_timer += 1;
    tl_metrics.operations += 1;

    query.for_each_mut(|(mut pos, vel)| {
        pos.x += vel.x;
        pos.y += vel.y;
    });
}

fn spatial_system(
    query_a: Query<(Entity, &Position, Option<&Player>), Without<Enemy>>,
    query_b: Query<(Entity, &Position), With<Enemy>>,
    mut writer: MessageWriter<CollisionMessage>,
) {
    query_a.for_each(|(ent_a, pos_a, player_opt)| {
        if player_opt.is_some() {
            for (ent_b, pos_b) in query_b.iter() {
                if (pos_a.x - pos_b.x).abs() < 5.0 {
                    writer.write(CollisionMessage {
                        entity_a: ent_a,
                        entity_b: ent_b,
                    });
                }
            }
        }
    });
}

fn logger_system(query: Query<&Position, Changed<Position>>) {
    query.for_each(|pos| {
        let _ = pos.x;
    });
}

fn damage_system(
    mut reader: MessageReader<CollisionMessage>,
    mut query: Query<&mut Health>,
    commands: Commands,
) {
    for msg in reader.iter() {
        if let Some(mut health) = query.get_mut(msg.entity_a) {
            health.0 -= 10.0;
            if health.0 <= 0.0 {
                commands.trigger(msg.entity_a, DestroyEvent { reason: 1 });
            }
        }

        commands.insert_component(msg.entity_b, Velocity { x: -1.0, y: -1.0 });
        commands.remove_component::<Player>(msg.entity_a);
    }
}

fn destroy_observer(trigger: Trigger<DestroyEvent>, query: Query<&Position>, commands: Commands) {
    if let Some(pos) = query.get(trigger.entity) {
        let _ = pos.x * (trigger.event.reason as f32);
    }

    commands.execute(|world| {
        if let Some(mut state) = world.resources_mut().get_mut::<GameState>() {
            state.score += 100;
        }
    });

    commands.despawn(trigger.entity);
}

struct CorePlugin;

impl Plugin for CorePlugin {
    fn build(&self, app: &mut App) {
        app.world.resources_mut().insert(GameState::default());
        app.world
            .thread_local_res_mut()
            .insert(ThreadMetrics::default());

        app.world.add_observer(destroy_observer);

        app.add_systems((
            spawn_system,
            movement_system.after(spawn_system),
            spatial_system.after(movement_system),
            damage_system.after(spatial_system),
            logger_system.after(movement_system),
        ));
    }
}

fn main() {
    let mut app = App::new();

    app.add_plugin(CorePlugin);

    for _ in 0..100000 {
        app.schedule.update(&mut app.world)
    }

    let manual_query = app.world.query::<(&Position, &Health), With<Player>>();
    for (pos, health) in manual_query.iter() {
        let _ = pos.x + health.0;
    }
}
